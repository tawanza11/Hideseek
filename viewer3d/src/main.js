import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './style.css';

const $ = (id) => document.getElementById(id);
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x101d2b);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101d2b);
scene.fog = new THREE.Fog(0x101d2b, 18, 34);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 5;
controls.maxDistance = 32;
controls.maxPolarAngle = Math.PI * 0.48;
scene.add(new THREE.AmbientLight(0xb6d3ec, 1.1));
const sun = new THREE.DirectionalLight(0xfff5dd, 2.3);
sun.position.set(-4, 12, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -12;
sun.shadow.camera.right = 12;
sun.shadow.camera.top = 12;
sun.shadow.camera.bottom = -12;
scene.add(sun);

const board = new THREE.Group();
const objects = new THREE.Group();
scene.add(board, objects);
const material = {
  tileA: new THREE.MeshStandardMaterial({ color: 0x263f52, roughness: 0.9 }),
  tileB: new THREE.MeshStandardMaterial({ color: 0x2a4659, roughness: 0.9 }),
  wall: new THREE.MeshStandardMaterial({ color: 0x667d8e, roughness: 0.75 }),
  lowWall: new THREE.MeshStandardMaterial({ color: 0xdfab67, roughness: 0.7 }),
  block: new THREE.MeshStandardMaterial({ color: 0xc1804f, roughness: 0.68 }),
  ramp: new THREE.MeshStandardMaterial({ color: 0x84ccc2, roughness: 0.65 }),
  seeker: new THREE.MeshStandardMaterial({ color: 0x43a5ff, metalness: 0.05, roughness: 0.35 }),
  hider: new THREE.MeshStandardMaterial({ color: 0x5bd991, metalness: 0.05, roughness: 0.4 }),
  sight: new THREE.LineBasicMaterial({ color: 0xffdf72 }),
};
const tileGeometry = new THREE.BoxGeometry(0.96, 0.1, 0.96);
const wallGeometry = new THREE.BoxGeometry(0.94, 1.25, 0.94);
const lowWallGeometry = new THREE.BoxGeometry(0.9, 0.5, 0.9);
const blockGeometry = new THREE.BoxGeometry(0.72, 0.72, 0.72);
const rampGeometry = new THREE.BoxGeometry(0.72, 0.15, 0.75);
const orbGeometry = new THREE.SphereGeometry(0.27, 24, 16);

function mesh(geometry, mat, x, y, z, group) {
  const item = new THREE.Mesh(geometry, mat);
  item.position.set(x, y, z);
  item.castShadow = true;
  item.receiveShadow = true;
  group.add(item);
  return item;
}
const seekerMesh = mesh(orbGeometry, material.seeker, 0, 0.44, 0, scene);
const hiderMesh = mesh(orbGeometry, material.hider, 0, 0.44, 0, scene);
function marker(letter, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext('2d');
  context.fillStyle = color;
  context.beginPath();
  context.arc(64, 64, 54, 0, Math.PI * 2);
  context.fill();
  context.lineWidth = 7;
  context.strokeStyle = '#ffffff';
  context.stroke();
  context.fillStyle = '#082030';
  context.font = 'bold 62px sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(letter, 64, 68);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas), transparent: true,
    depthTest: false, depthWrite: false,
  }));
  sprite.scale.set(0.72, 0.72, 1);
  sprite.renderOrder = 10;
  scene.add(sprite);
  return sprite;
}
const seekerMarker = marker('S', '#55aaff');
const hiderMarker = marker('H', '#65de9a');
const seekerGlow = new THREE.PointLight(0x3899ef, 1.4, 2.5);
const hiderGlow = new THREE.PointLight(0x46d587, 1.2, 2.5);
scene.add(seekerGlow, hiderGlow);
let sightLine = new THREE.Line(new THREE.BufferGeometry(), material.sight);
scene.add(sightLine);

let replay;
let current = 0;
let playing = false;
let lastAdvance = 0;
let width = 0;
let height = 0;
let possibleLowWalls = [];

function cellPosition([x, y]) {
  return [x - (width - 1) / 2, y - (height - 1) / 2];
}

function isPosition(value, mapWidth = width, mapHeight = height) {
  return Array.isArray(value) && value.length === 2 && value.every(Number.isInteger)
    && value[0] >= 0 && value[0] < mapWidth && value[1] >= 0 && value[1] < mapHeight;
}

function positions(value, label, mapWidth, mapHeight) {
  if (!Array.isArray(value) || !value.every((point) => isPosition(point, mapWidth, mapHeight))) throw new Error(`${label} ต้องเป็นรายการพิกัด [x, y] ในแผนที่`);
  return value;
}

function validate(data) {
  if (!data || typeof data !== 'object' || data.schema_version !== 1) throw new Error('ไฟล์ replay ต้องใช้ schema_version 1');
  if (!Array.isArray(data.map_rows) || !data.map_rows.length || !data.map_rows.every((row) => typeof row === 'string' && row.length === data.map_rows[0].length && /^[.#]+$/.test(row))) {
    throw new Error('map_rows ต้องเป็นแถวแผนที่ . และ # ที่ยาวเท่ากัน');
  }
  const mapWidth = data.map_rows[0].length;
  const mapHeight = data.map_rows.length;
  if (!Array.isArray(data.frames) || !data.frames.length) throw new Error('replay ต้องมี frames อย่างน้อยหนึ่งก้าว');
  for (const [index, frame] of data.frames.entries()) {
    if (!frame || frame.step !== index || !isPosition(frame.seeker, mapWidth, mapHeight) || !isPosition(frame.hider, mapWidth, mapHeight) || typeof frame.visible !== 'boolean') {
      throw new Error(`ข้อมูลก้าว ${index} ไม่ครบหรือพิกัดไม่ถูกต้อง`);
    }
    for (const name of ['blocks', 'ramps', 'low_walls']) {
      positions(frame[name] ?? data[name] ?? [], `frames[${index}].${name}`, mapWidth, mapHeight);
    }
  }
  width = mapWidth;
  height = mapHeight;
}

function resetCamera() {
  controls.target.set(0, 0, 0);
  camera.position.set(width * 0.52, Math.max(width, height) * 1.65, height * 0.7);
  controls.update();
}

function clearGroup(group) {
  // Shared geometries and materials are retained for later frames.
  while (group.children.length) group.remove(group.children[0]);
}

function buildMap() {
  clearGroup(board);
  possibleLowWalls = [...new Map(
    replay.frames.flatMap((frame) => frame.low_walls ?? replay.low_walls ?? [])
      .map((point) => [point.join(','), point]),
  ).values()];
  const lowWallCells = new Set(possibleLowWalls.map(([x, y]) => `${x},${y}`));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [px, pz] = cellPosition([x, y]);
      mesh(tileGeometry, (x + y) % 2 ? material.tileA : material.tileB, px, -0.08, pz, board);
      if (replay.map_rows[y][x] === '#' && !lowWallCells.has(`${x},${y}`)) {
        mesh(wallGeometry, material.wall, px, 0.59, pz, board);
      }
    }
  }
  resetCamera();
}

function updateObjects(frame) {
  clearGroup(objects);
  const lowWalls = frame.low_walls ?? replay.low_walls ?? [];
  const blocks = frame.blocks ?? replay.blocks ?? [];
  const ramps = frame.ramps ?? replay.ramps ?? [];
  const activeLowWalls = new Set(lowWalls.map(([x, y]) => `${x},${y}`));
  for (const point of possibleLowWalls) {
    if (!activeLowWalls.has(`${point[0]},${point[1]}`)) {
      const [x, z] = cellPosition(point);
      mesh(wallGeometry, material.wall, x, 0.59, z, objects);
    }
  }
  for (const point of lowWalls) {
    const [x, z] = cellPosition(point);
    mesh(lowWallGeometry, material.lowWall, x, 0.23, z, objects);
  }
  for (const point of blocks) {
    const [x, z] = cellPosition(point);
    mesh(blockGeometry, material.block, x, 0.38, z, objects);
  }
  for (const point of ramps) {
    const [x, z] = cellPosition(point);
    const item = mesh(rampGeometry, material.ramp, x, 0.18, z, objects);
    item.rotation.x = -0.35;
  }
  $('objects').textContent = `${blocks.length} / ${ramps.length}${lowWalls.length ? ` · กำแพงเตี้ย ${lowWalls.length}` : ''}`;
}

function setFrame(index) {
  current = Math.max(0, Math.min(index, replay.frames.length - 1));
  const frame = replay.frames[current];
  updateObjects(frame);
  const [sx, sz] = cellPosition(frame.seeker);
  const [hx, hz] = cellPosition(frame.hider);
  const overlap = sx === hx && sz === hz;
  seekerMesh.position.set(sx + (overlap ? -0.16 : 0), 0.37, sz);
  hiderMesh.position.set(hx + (overlap ? 0.16 : 0), 0.37, hz);
  seekerMarker.position.set(seekerMesh.position.x, 1.62, sz);
  hiderMarker.position.set(hiderMesh.position.x, 1.62, hz);
  seekerGlow.position.copy(seekerMesh.position);
  hiderGlow.position.copy(hiderMesh.position);
  sightLine.visible = frame.visible && !overlap;
  sightLine.geometry.dispose();
  sightLine.geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(sx, 0.42, sz), new THREE.Vector3(hx, 0.42, hz),
  ]);
  $('seeker-pos').textContent = `(${frame.seeker.join(', ')})`;
  $('hider-pos').textContent = `(${frame.hider.join(', ')})`;
  $('step').textContent = `${frame.step} / ${replay.max_steps ?? replay.frames.length - 1}`;
  $('action').textContent = ({ up: '↑ ขึ้น', down: '↓ ลง', left: '← ซ้าย', right: '→ ขวา' })[frame.action] ?? frame.action ?? 'เริ่มเกม';
  $('sight').textContent = frame.visible ? '◉ ฝ่ายหามองเห็นฝ่ายซ่อน' : '◌ ฝ่ายหามองไม่เห็นฝ่ายซ่อน';
  $('sight').classList.toggle('yes', frame.visible);
  const outcome = $('outcome');
  outcome.className = 'outcome';
  if (frame.event === 'seeker_captured' || (current === replay.frames.length - 1 && replay.outcome === 'seeker_captured')) {
    outcome.textContent = 'ฝ่ายหาจับฝ่ายซ่อนได้';
    outcome.classList.add('capture');
  } else if (frame.event === 'hider_survived' || (current === replay.frames.length - 1 && replay.outcome === 'hider_survived')) {
    outcome.textContent = 'ฝ่ายซ่อนรอดจนหมดเวลา';
    outcome.classList.add('survive');
  } else {
    const eventText = {
      seeker_block_pushed: 'ฝ่ายหาผลักบล็อก',
      hider_block_pushed: 'ฝ่ายซ่อนผลักบล็อก',
      seeker_climbed: 'ฝ่ายหาปีนข้ามกำแพง',
      hider_climbed: 'ฝ่ายซ่อนปีนข้ามกำแพง',
    };
    outcome.textContent = eventText[frame.event] ?? (frame.event && frame.event !== 'start' ? frame.event : 'เกมกำลังดำเนินอยู่');
  }
  $('timeline').value = current;
  $('progress').textContent = `${current} / ${replay.frames.length - 1}`;
}

function setPlaying(value) {
  playing = value;
  lastAdvance = performance.now();
  $('play').textContent = playing ? '❚❚ หยุด' : '▶ เล่น';
}

function loadReplay(data) {
  validate(data);
  replay = data;
  setPlaying(false);
  buildMap();
  $('map-title').textContent = typeof replay.map === 'string' ? replay.map : 'แผนที่';
  $('model-meta').textContent = `Seeker: ${replay.model ?? 'ไม่ระบุ'} · Hider: ${replay.hider_model ?? 'ไม่ระบุ'} · seed ${replay.episode_seed ?? '–'}`;
  $('timeline').max = replay.frames.length - 1;
  setFrame(0);
  $('error').hidden = true;
}

function showError(error) {
  $('error').textContent = error instanceof Error ? error.message : String(error);
  $('error').hidden = false;
}

$('file').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try { loadReplay(JSON.parse(await file.text())); } catch (error) { showError(error); }
  event.target.value = '';
});
$('play').addEventListener('click', () => {
  if (!replay) return;
  if (!playing && current === replay.frames.length - 1) setFrame(0);
  setPlaying(!playing);
});
$('timeline').addEventListener('input', (event) => {
  if (!replay) return;
  setPlaying(false);
  setFrame(Number(event.target.value));
});
$('reset-camera').addEventListener('click', resetCamera);

function resize() {
  const bounds = stage.getBoundingClientRect();
  renderer.setSize(bounds.width, bounds.height);
  camera.aspect = bounds.width / bounds.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

function animate(now) {
  requestAnimationFrame(animate);
  if (playing && replay && now - lastAdvance >= 550 / Number($('speed').value)) {
    if (current < replay.frames.length - 1) setFrame(current + 1);
    else setPlaying(false);
    lastAdvance = now;
  }
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

const replayPath = new URLSearchParams(window.location.search).get('replay') ?? '/replay-example.json';
fetch(replayPath)
  .then((response) => { if (!response.ok) throw new Error('ไม่พบ replay ตัวอย่าง'); return response.json(); })
  .then(loadReplay)
  .catch(showError);
