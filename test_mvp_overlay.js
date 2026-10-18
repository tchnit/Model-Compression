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

// Run the real event handlers with a video frame callback ahead of the playback clock.
const mvpSource = fs.readFileSync(`${__dirname}/demo_data/mvp.js`, 'utf8');
const mediaSource = mvpSource.slice(mvpSource.indexOf('  function media('), mvpSource.indexOf('  function preview('));
const selectionSource = mvpSource.slice(mvpSource.indexOf('  function selectBox('), mvpSource.indexOf('  function table('));
for (const frameCallbacks of [true, false]) {
  let callback, tableIndex, focused = false;
  const video = {tagName: 'VIDEO', currentTime: 0, duration: .45, paused: true, seeking: false,
    pause() { this.paused = true; }};
  if (frameCallbacks) video.requestVideoFrameCallback = handler => { callback = handler; return 1; };
  const nodes = new Map();
  const item = {video: true, source: 'fixture.webm', frames: variable, report: {source_fps: 10}, job: {status: 'cancelled'}};
  const playback = vm.createContext({
    player: null, videoCallback: null, currentFrame: null, tableFrame: null, selectedBox: -1, selected: 0, items: [item], mvpFrameAtTime,
    document: {createElement: () => video},
    $: id => {
      if (!nodes.has(id)) nodes.set(id, {append() {}});
      return nodes.get(id);
    },
    clearPlayer() { playback.player = playback.currentFrame = null; playback.selectedBox = -1; },
    renderOverlay() { focused = false; },
    table(boxes, index) { focused = false; tableIndex = index; playback.tableFrame = playback.currentFrame; },
    highlightSelection() {}
  });
  vm.runInContext(mediaSource + selectionSource, playback);
  playback.media(item);
  video.onloadedmetadata();
  video.paused = false; video.currentTime = .3;
  if (frameCallbacks) callback(0, {mediaTime: .35}); else video.ontimeupdate();
  video.ontimeupdate();
  const selectedFrame = playback.currentFrame;
  assert.equal(selectedFrame, variable[frameCallbacks ? 2 : 1], 'Playback clock events do not replace the displayed frame');
  assert.equal(tableIndex, selectedFrame.index, 'The table follows the displayed overlay frame');
  focused = true;
  playback.selectBox(0);
  assert.equal(video.currentTime, selectedFrame.source_time_seconds, 'Selection seeks to the source timestamp');
  video.onpause(); video.ontimeupdate(); video.onseeked();
  if (frameCallbacks) callback(0, {mediaTime: 0});
  assert.equal(playback.currentFrame, selectedFrame, 'A late callback must not replace the selected frame');
  assert.equal(playback.selectedBox, 0, 'Pause and seek events keep the selected box');
  assert.equal(focused, true, 'Unchanged frames keep the focused SVG or table row');
  playback.selectBox(0);
  assert.equal(playback.selectedBox, -1, 'Selecting the same box again deselects it');
  playback.selectBox(0, 0);
  video.onseeked();
  assert.equal(playback.currentFrame, variable[0]);
  assert.equal(tableIndex, 0, 'Selecting another frame updates the table');
  assert.equal(playback.selectedBox, 0);
  video.seeking = true; video.currentTime = .5;
  video.onpause(); video.ontimeupdate();
  if (frameCallbacks) callback(0, {mediaTime: .35});
  assert.equal(playback.currentFrame, variable[0], 'Pending seeks ignore old playback events');
  video.seeking = false; video.onseeked();
  assert.equal(playback.currentFrame, null, 'Seeking past processed frames clears the overlay');
  assert.equal(playback.selectedBox, -1);
}
console.log('MVP overlay: frame timing, selection through pause/seek, focus and visibility controls PASS');
