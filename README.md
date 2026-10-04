# Hide-and-Seek RL

เกมซ่อนหาแบบตาราง 2D สำหรับทดลอง Reinforcement Learning: **Seeker เรียนรู้ด้วย PPO** เพื่อจับ Hider ที่เดินแบบสุ่ม ใน V1 มีผู้เล่นฝ่ายละ 1 ตัว เดินขึ้น/ลง/ซ้าย/ขวา และไม่มีสิ่งกีดขวางภายในฉาก

## เริ่มใช้งาน

ใช้ Python 3.10 ขึ้นไป (ทดสอบการพัฒนาด้วย Python 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m training.train --timesteps 100000
python -m evaluation.evaluate --episodes 100
python -m evaluation.evaluate --episodes 10 --watch
```

คำสั่ง `--watch` เปิดหน้าต่างเกมเพื่อดู **รอบแรกของ Seeker ที่ฝึกแล้ว** จึงต้องใช้เครื่องที่มีหน้าจอ กดปิดหน้าต่างเพื่อหยุดดู โปรแกรมบันทึกโมเดลใน `models/seeker_ppo.zip`, ค่าตั้งต้นใน `models/seeker_ppo.json`, ผลฝึกใน `results/train_episodes.csv`, ผลประเมินใน `results/evaluation.csv` และกราฟใน `results/metrics.png` รีโปมีผลฝึกตัวอย่างที่รันแล้ว จึงสามารถข้ามคำสั่งฝึกและเริ่มจากคำสั่งประเมินหรือดูเกมได้ทันที

ปรับขนาดเกมและเวลาฝึกได้ เช่น:

```bash
python -m training.train --grid-size 8 --max-steps 64 --timesteps 200000 --seed 42
```

## กติกา V1

- พื้นที่ `8 × 8` ช่องโดยค่าเริ่มต้น จุดเริ่ม Seeker/Hider ไม่ซ้ำกัน
- การกระทำของ Seeker: `0=ขึ้น`, `1=ลง`, `2=ซ้าย`, `3=ขวา`; เดินชนขอบแล้วอยู่ที่เดิม
- Hider เดินสุ่ม 1 ช่องที่ถูกต้องหลัง Seeker; หาก Seeker ไปถึงช่อง Hider ก่อน จะจบทันที
- สิ่งที่ Seeker เห็น: ตำแหน่งตัวเอง ตำแหน่ง Hider และระยะถึงขอบ 4 ด้าน (รวม 8 ค่า ปรับสเกลเป็น 0–1)
- รางวัล: จับได้ `+1`, ครบ 64 ก้าวแล้วยังจับไม่ได้ `-1`, ก้าวอื่น `-0.01`
- ประเมินด้วยชุด seed เดียวกันเทียบ **PPO** กับ Seeker ที่เดินสุ่ม โดยรายงานอัตราจับได้

ข้อมูลและผลทดลอง V1 อยู่ใน [docs/report.md](docs/report.md)

## V2: ฝึก Hider และ Seeker สลับกัน

V2 ให้ Hider เรียนรู้การหนีจาก Seeker V1 ที่ตรึงไว้ก่อน แล้วฝึก Seeker ต่อกับ Hider ที่ได้ สลับกัน 2 รอบ โดยคู่แข่งจะหยุดเรียนรู้ระหว่างที่อีกฝ่ายกำลังฝึก V1 ยังทำงานแบบเดิมและโมเดล V1 ไม่ถูกเขียนทับ

```bash
python -m training.train_v2
python -m evaluation.evaluate_v2
python -m evaluation.evaluate_v2 --watch
```

รีโปมีโมเดล V2 และผลประเมินตัวอย่างแล้ว จึงรันคำสั่งประเมินได้ทันที `--watch` แสดงเกม V2 Seeker พบ V2 Hider 1 รอบ การฝึกใหม่จะเขียนทับไฟล์ `models/hider_v2.*`, `models/seeker_v2.*` และบันทึกผลใน `results/v2_*` โดยค่าเริ่มต้นประเมินคู่แข่ง 6 แบบ กลุ่ม seed 1000, 2000, 3000 กลุ่มละ 100 episode

รายละเอียดวิธีฝึก ผลรายคู่ และข้อจำกัดอยู่ใน [docs/v2_report.md](docs/v2_report.md)

## V3: ห้อง กำแพง และการบังสายตา

V3 ใช้แผนที่ 10 × 10 ที่มีห้องและทางเชื่อม กำแพงขวางการเดินและการมองเห็น แต่ละฝ่ายเห็นตำแหน่งคู่แข่งเฉพาะเมื่อมีแนวสายตาถึงกัน หากมองไม่เห็น ตำแหน่งคู่แข่งใน observation จะเป็นศูนย์พร้อมธง `visible=0` ทั้งสองฝ่ายรู้แผนที่กำแพง ค่าตั้งต้นให้เวลา 100 ก้าวต่อเกม

```bash
python -m training.train_v3
python -m evaluation.evaluate_v3
python -m evaluation.evaluate_v3 --watch
```

การฝึกเริ่มโมเดลใหม่สำหรับ 3 training seed (41, 42, 43) โดยฝึก Seeker กับ Hider สุ่ม 100,000 ก้าวก่อน แล้วฝึก Hider/Seeker สลับกัน 2 รอบ รอบละฝ่ายละ 50,000 ก้าว คู่แข่งหยุดเรียนรู้ในแต่ละช่วง ประเมิน 3 กลุ่ม seed ที่แยกจากการฝึก กลุ่มละ 100 เกมต่อคู่ต่อ training seed; `--watch` เปิดภาพเกมตัวอย่าง 1 รอบ โมเดล V1/V2 ไม่ถูกเขียนทับ รายละเอียดและผลจริงอยู่ใน [docs/v3_report.md](docs/v3_report.md)

ดูภาพเกมจากโมเดลที่ฝึกแล้วได้ที่ [Seeker จับได้](results/v3_demo_capture.gif) และ [Hider รอดครบเวลา](results/v3_demo_hidden.gif) สีน้ำเงินคือ Seeker, สีเขียวคือ Hider, สีเทาคือกำแพง ข้อความบนภาพบอกว่า Seeker มองเห็น Hider หรือไม่ หากต้องการเปิดเกมสดหลายก้าวโดยไม่เขียนทับผลประเมินเดิม:

```bash
python -m evaluation.evaluate_v3 --training-seeds 41 --eval-seed-starts 10014 --episodes-per-group 1 --output-dir /tmp/hideseek-v3-preview --fps 2 --watch
```

## V3.1: Seeker เลือกทิศที่เดินได้

V3.1 วัดพบว่า Seeker V3 เลือกเดินชนกำแพงบ่อย จึงฝึกต่อจากโมเดล V3 โดยเทียบชุดที่มีค่าปรับกำแพงกับชุด Control ที่ไม่มีค่าปรับ ผลยืนยันพบว่า **ค่าปรับไม่ช่วยสม่ำเสมอ** แต่การใช้ Control ร่วมกับการเลือก action ที่มีความน่าจะเป็นสูงสุดเฉพาะทิศที่เดินได้ จับ Hider V3 ได้ 72.3% บน eval seed ชุดใหม่ 900 เกม ตัวกรองใช้เพียงตำแหน่งตัวเองและแผนที่กำแพงที่โมเดลเห็นอยู่แล้ว โมเดล V1–V3 ยังอยู่เหมือนเดิม

รีโปมีโมเดล V3.1 และผลประเมินแล้ว ดู [ตัวอย่างเกม](results/v31_demo_capture.gif) หรือเปิดเกมสดได้ทันที:

```bash
python -m evaluation.watch_v31
```

หากต้องการฝึกและประเมินซ้ำ:

```bash
python -m training.train_v31
python -m evaluation.evaluate_v31
python -m evaluation.evaluate_v31 --eval-seed-starts 70000 80000 90000 --legal-mask
```

วิธีทดลอง ผลราย seed และข้อจำกัดอยู่ใน [docs/v31_report.md](docs/v31_report.md)
