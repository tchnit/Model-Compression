'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {mvpFrameAtTime} = require('./demo_data/mvp.js');

const frames = [0, 1, 2].map(index => ({index, time_seconds: index / 10}));
assert.equal(mvpFrameAtTime(frames, 0, 10), frames[0]);
assert.equal(mvpFrameAtTime(frames, .19, 10), frames[1]);
assert.equal(mvpFrameAtTime(frames, .29, 10), frames[2]);
assert.equal(mvpFrameAtTime(frames, .31, 10), null, 'Never reuse the last box on unprocessed frames');
assert.equal(mvpFrameAtTime(frames, .3, 10), null, 'Floating point frame boundaries do not leak boxes');
assert.equal(mvpFrameAtTime(frames, -.01, 10), null);
assert.equal(mvpFrameAtTime([], 0, 10), null);
assert.equal(mvpFrameAtTime(frames, NaN, 10), null);
assert.equal(mvpFrameAtTime(frames, 0, 0), null);
const variable = [0, .12, .35].map((source_time_seconds, index) => ({index, source_time_seconds, time_seconds: index / 10}));
assert.equal(mvpFrameAtTime(variable, .3, 10), variable[1], 'Use source timestamps for variable frame rate video');
assert.equal(mvpFrameAtTime(variable, .36, 10), variable[2]);

// Exercise the renderer shared by Studio, Hardware and MVP without a browser dependency.
const html = fs.readFileSync('demo.html', 'utf8');
const source = html.slice(html.indexOf('function generateSvgElements('), html.indexOf('function buildBDD100KJson('));
const render = vm.runInNewContext(`const CLASS_COLORS = {car:'#06b6d4'}; ${source}; generateSvgElements;`);
const boxes = [{class_name: 'car', conf: .85, box_px: [10, 20, 110, 90]}];
const defaults = render(boxes, 640, 360);
assert.match(defaults, /class="bbox-rect"/);
assert.match(defaults, />car 85%<\/text>/);
const scoresOnly = render(boxes, 640, 360, 1, {boxes: false, labels: false, scores: true, interactive: true});
assert.doesNotMatch(scoresOnly, /class="bbox-rect"/);
assert.match(scoresOnly, />85%<\/text>/);
assert.match(scoresOnly, /role="button"/);
assert.match(scoresOnly, /data-box-index="0"/);
const boxesOnly = render(boxes, 640, 360, 1, {labels: false, scores: false});
assert.match(boxesOnly, /class="bbox-rect"/);
assert.doesNotMatch(boxesOnly, /<text/);
console.log('MVP overlay: frame boundaries, source timestamps, interactive boxes and visibility controls PASS');
