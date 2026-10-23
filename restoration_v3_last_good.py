import numpy as np, cv2
from scipy import ndimage
from skimage.morphology import remove_small_objects, reconstruction, disk
from skimage.filters import threshold_otsu, apply_hysteresis_threshold


def _ellipse(k):
	k = int(k) | 1
	return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def estimate_stroke_radius(binary_ink, max_r=12):
	"""Granulometry (unchanged idea): erode with growing disks; the radius where the
	ink-loss curve flattens is ~ the dominant pen-stroke half-width."""
	prev = int(binary_ink.sum())
	if prev == 0:
		return 2
	losses = []
	b = binary_ink.astype(np.uint8)
	for r in range(1, max_r + 1):
		cur = int(cv2.erode(b, _ellipse(2 * r + 1)).sum())
		losses.append(prev - cur)
		prev = cur
	losses = np.array(losses, dtype=float)
	if losses.max() <= 0:
		return 2
	r = int(np.argmax(losses / losses.max() < 0.15)) + 1
	return max(2, min(r, max_r))


def flatten_illumination(gray, stroke_r):
	"""Divide by a closing-based paper estimate. Kernel scales with stroke width (a closing
	kernel must exceed the widest ink stroke or the ink leaks into the 'paper' estimate),
	then the estimate is smoothed so no blocky closing artefacts get divided back in."""
	k = max(25, 8 * stroke_r + 1)
	bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _ellipse(k))
	bg = cv2.GaussianBlur(bg, (0, 0), k / 6.0)
	return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def text_scale(binary, stroke_r):
	"""Typical size of one connected mark on the page (letter in print, word in cursive): the median
	long side of the components' bounding boxes. Gives every 'is this long enough to be a ruled
	line?' test a page-relative yardstick instead of a fixed pixel count."""
	lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
	if n == 0:
		return 4 * stroke_r
	sizes = []
	for sl in ndimage.find_objects(lab):
		hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
		if max(hh, ww) >= 3 * stroke_r:       # ignore specks and dots
			sizes.append(max(hh, ww))
	return float(np.median(sizes)) if sizes else 4.0 * stroke_r


def _band_beside(m, pad, side, horizontal, sign):
	"""Pixels in a band of width `side`, starting `pad` pixels away from run mask m, on ONE side of it
	(sign=+1: below / right, -1: above / left)."""
	k = pad + side
	if horizontal:
		kern = np.ones((k + 1, 1), np.uint8)
		anchor = (0, 0) if sign > 0 else (0, k)
	else:
		kern = np.ones((1, k + 1), np.uint8)
		anchor = (0, 0) if sign > 0 else (k, 0)
	outer = cv2.dilate(m, kern, anchor=anchor)
	kin = pad
	if horizontal:
		kern_i = np.ones((kin + 1, 1), np.uint8)
		anchor_i = (0, 0) if sign > 0 else (0, kin)
	else:
		kern_i = np.ones((1, kin + 1), np.uint8)
		anchor_i = (0, 0) if sign > 0 else (kin, 0)
	inner = cv2.dilate(m, kern_i, anchor=anchor_i)
	return (outer > 0) & ~(inner > 0)


def remove_lines(binary, stroke_r, scale):
	"""Ruled / margin / frame lines, solid or dashed.
	* LONG is measured against this page's own text size (several text-marks long), so a tall letter stem
	  (l, f, t) is never mistaken for a rule, even in tiny print.
	* THIN: ink thinner than a pen stroke.
	* ISOLATED (needed to catch DASHED lines): after bridging small gaps, a rule is a narrow run with
	  almost no other ink right beside it; a row of text always has ink in the neighbouring rows, so a
	  bridged text row never qualifies no matter how long or thin its strokes are.
	Only thin ink on the run is removed, so a letter crossing a rule keeps its body."""
	b = (binary > 0).astype(np.uint8)
	L = int(max(25, 10 * stroke_r, 4 * scale))
	thin_cap = max(3.0, stroke_r * 0.8)
	thick = cv2.distanceTransform(b, cv2.DIST_L2, 5) * 2
	thin = (thick <= thin_cap) & (b > 0)

	# solid thin runs
	solid = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
	solid |= cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
	out = (solid > 0) & thin

	# dashed, isolated runs
	gap = 2 * stroke_r + 3
	pad = int(thin_cap) + 1
	side = max(4, int(scale) // 2)                 # how far beside the run we look for neighbouring ink
	for horizontal in (True, False):
		gk, lk = ((gap, 1), (L, 1)) if horizontal else ((1, gap), (1, L))
		bridged = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, gk))
		run = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, lk))
		lab, n = ndimage.label(run, structure=np.ones((3, 3)))
		if n == 0:
			continue
		# np.ones shape is (rows, cols): look ABOVE/BELOW a horizontal run, LEFT/RIGHT of a vertical one
		near_k = (2 * pad + 1, 1) if horizontal else (1, 2 * pad + 1)
		far_k = (2 * (pad + side) + 1, 1) if horizontal else (1, 2 * (pad + side) + 1)
		for i, sl in enumerate(ndimage.find_objects(lab), 1):
			y0, y1 = max(sl[0].start - side - pad, 0), min(sl[0].stop + side + pad, b.shape[0])
			x0, x1 = max(sl[1].start - side - pad, 0), min(sl[1].stop + side + pad, b.shape[1])
			m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
			length = max(sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)
			if m.sum() / length > min(thin_cap + 2, 0.25 * scale):    # not narrow (relative to text size)
				continue
			inkl = b[y0:y1, x0:x1] > 0
			clear_both_sides = True
			for sign in (1, -1):                   # the two sides of the run, tested separately
				band = _band_beside(m, pad, side, horizontal, sign)
				if (inkl & band).sum() > 0.15 * max(band.sum(), 1):
					clear_both_sides = False
					break
			if clear_both_sides:
				out[y0:y1, x0:x1] |= (m > 0) & thin[y0:y1, x0:x1]
	return out


def separate_blobs(binary, stroke_r, max_component_area):
	"""Thick compact blobs (stamps, seals, blots): opening-by-reconstruction with a disk wider
	than any pen stroke, plus a shape-gated size cap."""
	b = (binary > 0).astype(np.uint8)
	marker = cv2.erode(b, disk(max(3, stroke_r * 4)).astype(np.uint8))
	blobs = reconstruction(marker, b, method="dilation").astype(bool)
	lab, n = ndimage.label(binary)
	if n:
		objs = ndimage.find_objects(lab)
		areas = ndimage.sum(binary, lab, range(1, n + 1))
		ids = []
		for i, sl in enumerate(objs, 1):
			a = areas[i - 1]
			if a <= max_component_area:
				continue
			h = sl[0].stop - sl[0].start
			w = sl[1].stop - sl[1].start
			if a / (h * w) > 0.35 and max(h, w) / max(1, min(h, w)) < 4:
				ids.append(i)
		if ids:
			blobs |= np.isin(lab, ids)
	return blobs


def sharpness_mask(norm, stroke_r, thr=0.16):
	"""Bleed-through ('show-through' ink from the other side) is blurred by the paper, real ink is
	sharp. Around each pixel compare the strongest edge to the strongest darkness in a
	stroke-sized window: a step edge of height C blurred by sigma has max gradient ~ 0.4*C/sigma,
	so sharp ink gives a high ratio and blurry bleed a low one. Scale-free, no per-dataset tuning."""
	f = norm.astype(np.float32)
	g = cv2.GaussianBlur(f, (0, 0), 1.0)
	gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
	gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
	grad = cv2.magnitude(gx, gy)
	dark = np.maximum(255.0 - g, 0)
	g_max = cv2.dilate(grad, _ellipse(2 * (stroke_r + 2) + 1))   # edges anywhere within a stroke width
	d_max = cv2.dilate(dark, _ellipse(5))                         # but darkness measured locally, so a faint
	                                                              # hairline isn't judged against a thick stem beside it
	ratio = g_max / np.maximum(d_max, 1.0)
	return (ratio >= thr) & (d_max > 8), ratio


def adaptive_sharp_thr(ratio, core, cap, frac=0.5):
	"""A fraction of how sharp this page's REAL (dark-core) ink is, capped. Using the dark core as the
	reference keeps bleed-through from dragging the reference down, while an inherently soft scan
	(photo, blurry microfilm) still gets a proportionally lower bar instead of being read as all-bleed."""
	vals = ratio[core]
	if vals.size == 0:
		return cap
	return float(np.clip(frac * np.percentile(vals, 90), 0.04, cap))


def filter_components(binary, dark, ratio, stroke_r):
	"""Per-component clean-up, all relative to THIS page's own ink (no absolute thresholds):
	* faint components -- much lighter than the page's real ink -- are bleed / stamp ghosts / stains
	* low-sharpness components are blurry show-through
	Dark, sharp specks (dots, diacritics, punctuation) are always kept."""
	lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
	if n == 0:
		return binary
	idx = np.arange(1, n + 1)
	area = ndimage.sum(binary, lab, idx)
	dmax = ndimage.maximum(dark, lab, idx)
	# sharpness is judged on each component's boundary pixels (that's where edges are; interiors of
	# thick strokes have no gradient and would read as 'blurry')
	bnd = binary & ~cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
	rmean = ndimage.mean(ratio, lab * bnd, idx)
	big = area >= 4 * np.pi * stroke_r ** 2
	ref = np.percentile(dmax, 95)           # how dark this page's REAL ink gets
	faint = dmax < 0.5 * ref
	ref_r = np.percentile(rmean[big], 75) if big.any() else np.percentile(rmean, 75)
	blurry = rmean < 0.45 * ref_r   # relative to this page's own sharpest ink
	small = area < np.pi * stroke_r ** 2      # only truly tiny specks; hairline strokes of small print are bigger than this
	# faint comps are bleed / stains / grain UNLESS they are both crisp-edged and big enough to be a stroke
	faint_and_soft = faint & ((rmean < 0.8 * ref_r) | small)
	drop = faint_and_soft | blurry
	keep_ids = np.concatenate([[False], ~drop])
	return keep_ids[lab]


def remove_border_bars(binary, stroke_r):
	"""Without cropping, scan/page edges show up as long SOLID bars hugging the image border.
	Remove only components that touch the border AND are bar-shaped (very elongated and solid);
	handwriting that merely reaches the page edge is never that shape, so it stays."""
	h, w = binary.shape
	lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
	if n == 0:
		return binary
	out = binary.copy()
	for i, sl in enumerate(ndimage.find_objects(lab), 1):
		y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
		if not (y0 <= 1 or x0 <= 1 or y1 >= h - 1 or x1 >= w - 1):
			continue
		bh, bw = y1 - y0, x1 - x0
		long_side, short_side = max(bh, bw), max(1, min(bh, bw))
		if long_side < 20 * stroke_r or long_side / short_side < 10:
			continue
		area = (lab[sl] == i).sum()
		if area / (bh * bw) >= 0.5 or short_side <= 3 * stroke_r:
			out[sl][lab[sl] == i] = False
	return out


def scan_border_zone(gray, stroke_r):
	"""Dark scanner background / book edge / binding shadow beside the page. Defined by shape, not by
	position: a LARGE SOLID dark region that touches the image border. Text can't be that -- eroding
	the dark mask with a kernel wider than any stroke wipes strokes out but leaves such regions.
	Replaces blind cropping: nothing inside the page is ever cut, and text near a clean edge stays."""
	paper = float(np.percentile(gray, 90))
	dark = (gray < 0.7 * paper).astype(np.uint8)
	k = max(15, 6 * stroke_r + 1)
	core = cv2.erode(dark, _ellipse(k))
	if not core.any():
		return np.zeros(gray.shape, bool)
	grown = reconstruction(core, dark, method="dilation").astype(bool)
	lab, n = ndimage.label(grown)
	h, w = gray.shape
	zone = np.zeros((h, w), bool)
	for i in range(1, n + 1):
		m = lab == i
		# a scanner border / book edge hugs a good part of a whole image side; a stain or shadow
		# blob only touches the edge briefly (and may have text under it, which must survive)
		contact = max(m[0].sum() / w, m[-1].sum() / w, m[:, 0].sum() / h, m[:, -1].sum() / h)
		# ... AND on average far darker than paper (a scanner background is; a stain or shadow is not)
		if contact >= 0.5 and gray[m].mean() < 0.42 * paper:
			zone |= m
	if not zone.any():
		return zone
	# the band fades into the paper over a short distance: take the darker-than-paper pixels right next to it
	near = cv2.dilate(zone.astype(np.uint8), _ellipse(4 * stroke_r + 11)).astype(bool)
	zone |= near & (gray < 0.85 * paper)
	return cv2.dilate(zone.astype(np.uint8), _ellipse(2 * stroke_r + 5)).astype(bool)


def restore_document(img_path, bleed_thr=0.16, gamma=1.3, edge_frac=0.25, debug=False):
	st = {}
	gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if gray is None:
		return None
	h, w = gray.shape  # NOTE: no resize, no crop -- output is exactly the input size

	rough = flatten_illumination(gray, 2)
	_, rb = cv2.threshold(rough, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	stroke_r = estimate_stroke_radius(rb > 0)
	norm = flatten_illumination(gray, stroke_r)

	bh = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, _ellipse(max(3, 2 * stroke_r + 1)))
	enh = cv2.subtract(norm, bh)

	t = threshold_otsu(enh)
	strict = enh < t
	# ink-core darkness of THIS page. On pages with bleed-through Otsu lands inside the bleed, so the
	# thresholds are tied to the real ink core as well as to Otsu.
	lo = float(np.percentile(enh[strict], 1)) if strict.any() else 0.0
	t_hi = min(t, lo + 0.40 * (255 - lo))                           # a real-ink pixel
	t_tight = min(t + edge_frac * (255 - t), lo + 0.55 * (255 - lo))  # dark enough to be ink on its own
	t_loose = lo + 0.75 * (255 - lo)                                # faint ink -- only if its edges are crisp

	_, ratio = sharpness_mask(norm, stroke_r, bleed_thr)
	core_px = enh < t_hi
	sharp = ratio >= adaptive_sharp_thr(ratio, core_px, bleed_thr)
	# faint ink must be about as crisp as the page's real ink (bleed-through is much blurrier)
	crisp = ratio >= adaptive_sharp_thr(ratio, core_px, 1.0, frac=0.6)
	faint_ink = crisp & (enh < lo + 0.75 * (255 - lo))

	low = (enh < t_tight) | faint_ink
	core = (enh < t_hi) | faint_ink
	# Edges grow out of the real-ink core by a stroke-sized distance only. Unlimited hysteresis lets a whole
	# connected halo of faint bleed-through fuse onto the ink and survive; a bounded reach cannot.
	reach = stroke_r // 2 + 2
	binary = core | (cv2.dilate(core.astype(np.uint8), _ellipse(2 * reach + 1)).astype(bool) & low)
	st['hyst'] = binary.copy()
	binary &= (sharp | (enh < t * 0.6))        # blurry-but-dark pixels only if very dark
	st['gate'] = binary.copy()

	binary &= ~scan_border_zone(gray, stroke_r)
	binary = remove_border_bars(binary, stroke_r)

	lines = remove_lines(binary, stroke_r, text_scale(strict, stroke_r))
	lines = cv2.dilate(lines.astype(np.uint8), _ellipse(3)).astype(bool)
	text = binary & ~lines
	st['lines_removed'] = text.copy()

	blobs = separate_blobs(text, stroke_r, max_component_area=max(2500, 200 * stroke_r ** 2))
	text &= ~blobs
	st['blobs_removed'] = text.copy()

	closed = cv2.morphologyEx(text.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(3)).astype(bool)
	text = filter_components(closed, 255.0 - norm, ratio, stroke_r)   # un-boosted darkness: the boost saturates
	st['filtered'] = text.copy()
	text = remove_small_objects(text, max(3, stroke_r), connectivity=2)   # keep dots, full stops, diacritics
	st['small_removed'] = text.copy()
	filled = ndimage.binary_fill_holes(text)

	keep = cv2.dilate(filled.astype(np.uint8), _ellipse(3)).astype(bool) & (enh < t_loose)

	tone = np.clip((enh.astype(np.float32) - lo) / max(255.0 - lo, 1.0), 0, 1) ** gamma
	out = np.full((h, w), 255, np.uint8)
	out[keep] = (tone[keep] * 255).astype(np.uint8)
	if debug:
		return out, dict(stroke_r=stroke_r, t=t, lines=lines, blobs=blobs, sharp=sharp, ratio=ratio, binary=binary, filled=filled, norm=norm, enh=enh, st=st, t_hi=t_hi, t_tight=t_tight, t_loose=t_loose, strict=strict)
	return out
