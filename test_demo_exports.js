const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const element = () => ({children: [], classList: {add(){},remove(){}}, addEventListener(){},
  append(...children){this.children.push(...children)}, replaceChildren(...children){this.children=children}});
let exported, downloaded;
const context = vm.createContext({
  window: {addEventListener(){}}, console, Blob, setTimeout(){},
  URL: {createObjectURL(blob){exported=blob;return 'blob:fixture'}},
  document: {addEventListener(){},getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},
    createElement(tag){const el=element();el.tag=tag;el.click=()=>{downloaded=el.download};return el}}
});
for(const name of ['overlays.js','studio.js'])vm.runInContext(fs.readFileSync(`${__dirname}/demo_data/${name}`,'utf8'),context);
vm.runInContext(`demoData={scenes:[{id:'upload',file:'example.PNG',title:'" onload=alert(1) ',tag:'Upload',data_uri:'data:image/png;base64,AA'}],detections:{b:{upload:{boxes:[{class_id:2,class_name:'car',conf:.9,box_norm:[.1,.2,.3,.6]}]}}}};currentSceneId='upload';currentModelB='b';renderSceneList();downloadYoloTxt();`,context);
const card=elements.get('sceneListContainer').children[0];
assert.equal(card.children[0].alt,'" onload=alert(1) ');
assert.equal(card.children[0].onload,undefined);
assert.equal(card.children[1].children[0].textContent,'" onload=alert(1) ');
assert.equal(elements.get('sceneListContainer').innerHTML,undefined);
assert.equal(downloaded,'example.txt');
const yoloBlob=exported;
vm.runInContext(`updateExportJson([], demoData.scenes[0], 100, 100)`,context);
assert.deepEqual(JSON.parse(elements.get('jsonPreviewBox').textContent).attributes, {});
vm.runInContext(`demoData.scenes[0].attributes={weather:'rainy',timeofday:'night'};updateExportJson([], demoData.scenes[0], 100, 100)`,context);
assert.equal(JSON.parse(elements.get('jsonPreviewBox').textContent).attributes.timeofday, 'night');
assert.deepEqual(JSON.parse(vm.runInContext(`JSON.stringify(buildBDD100KJson([
 {class_name:'person',box_norm:[0,0,1,1],conf:.9},
 {class_name:'bike',box_norm:[0,0,1,1],conf:.8},
 {class_name:'motor',box_norm:[0,0,1,1],conf:.7}
],demoData.scenes[0],100,100).labels.map(b=>b.category))`,context)), ['pedestrian','bicycle','motorcycle']);
vm.runInContext('downloadBDD100KJson()',context);
assert.equal(downloaded,'prelabel_example_b.json');
yoloBlob.text().then(text=>{assert.equal(text.trim().split(/\s+/).length,5);assert.equal(text,'2 0.200000 0.400000 0.200000 0.400000');console.log('Upload title safety, BDD metadata/categories and five-column YOLO labels passed.');}).catch(error=>{console.error(error);process.exitCode=1});
