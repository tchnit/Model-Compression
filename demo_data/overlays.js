const CLASS_COLORS = {
  'car': '#06b6d4',          // Cyan
  'bus': '#3b82f6',          // Blue
  'truck': '#8b5cf6',        // Violet
  'person': '#10b981',       // Emerald
  'rider': '#34d399',        // Light Green
  'bike': '#14b8a6',         // Teal
  'motor': '#f97316',        // Orange
  'train': '#ec4899',        // Pink
  'traffic light': '#f59e0b', // Amber
  'traffic sign': '#ef4444'   // Red
};

function generateSvgElements(boxes, w, h, labelScale = 1, display = {}) {
  return boxes.map((b, index) => {
    let x, y, boxW, boxH;
    if (b.box_px && b.box_px.length === 4) {
      x = b.box_px[0];
      y = b.box_px[1];
      boxW = b.box_px[2] - b.box_px[0];
      boxH = b.box_px[3] - b.box_px[1];
    } else {
      const [nx1, ny1, nx2, ny2] = b.box_norm;
      x = nx1 * w;
      y = ny1 * h;
      boxW = (nx2 - nx1) * w;
      boxH = (ny2 - ny1) * h;
    }

    const color = display.colors?.[b.class_name] || CLASS_COLORS[b.class_name] || '#38bdf8';
    const confPct = Math.round(b.conf * 100);
    const label = [display.labels !== false ? b.class_name : '', display.scores !== false ? `${confPct}%` : ''].filter(Boolean).join(' ');
    const labelWidth = Math.min(w, (label.length * 7 + 8) * labelScale);
    const labelX = Math.max(0, Math.min(x, w - labelWidth));

    return `
      <g class="bbox-group" data-box-index="${index}" ${display.interactive ? `tabindex="0" role="button" aria-label="${b.class_name} ${confPct}%, đối tượng ${index + 1}" aria-pressed="false"` : ''}>
        <title>${b.class_name} (${confPct}%) [x:${Math.round(x)}, y:${Math.round(y)}, w:${Math.round(boxW)}, h:${Math.round(boxH)}]</title>
        ${display.boxes !== false ? `<rect class="bbox-rect" x="${x}" y="${y}" width="${boxW}" height="${boxH}" stroke="${color}" fill="${color}" stroke-opacity="0.9"/>` : ''}
        ${label ? `
        <rect x="${labelX}" y="${Math.max(0, y - 16 * labelScale)}" width="${labelWidth}" height="${16 * labelScale}" fill="${color}" fill-opacity="0.9" rx="3"></rect>
        <text class="bbox-label" style="font-size:${11 * labelScale}px" x="${labelX + 4 * labelScale}" y="${Math.max(12 * labelScale, y - 4 * labelScale)}" fill="#ffffff">${label}</text>` : ''}
      </g>
    `;
  }).join('');
}

if (typeof module !== 'undefined') module.exports = {generateSvgElements};
