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

## V4: ทดสอบบนแผนที่ใหม่

V4 เพิ่มแผนที่ฝึก T1/T2 และแยก E1–E3 สำหรับเลือกแนวทาง กับ F1/F2 สำหรับทดสอบครั้งสุดท้าย ทดลองฝึก Seeker ต่อจาก V3.1 แบบมีและไม่มีความจำช่องที่เคยเดิน โดยให้รางวัลสำรวจเหมือนกันทั้งสองแบบ

บนแผนที่ F1/F2 ที่กันไว้ V4 แบบไม่มีความจำซึ่งเลือกจากชุด E จับ Hider V3 ได้ **202/600 (33.7%)** เทียบกับ V3.1 **146/600 (24.3%)** และ Seeker สุ่ม **186/600 (31.0%)** แต่ยังต่ำกว่าตัวสุ่มบน F1 และบนแผนที่เดิมได้ 62.7% เทียบกับ V3.1 ที่ 72.0% จึงยังเป็นผลทดลอง ไม่ได้แทนโมเดล V3.1 แบบมีความจำได้ 201/600 (33.5%) บนชุด F จึงยังไม่มีหลักฐานว่าความจำเพิ่มอัตราจับโดยรวม

ดู [รายงาน V4](docs/v4_report.md), [กราฟชุด E](results/v4_models_validation_metrics.png) และ [กราฟชุด F](results/v4_models_test_metrics.png) โมเดลและผลรายเกมอยู่ใน `models/seeker_v4_*` และ `results/v4_*` ตามลำดับ หากต้องการรันทดลองซ้ำ:

```bash
python -m evaluation.evaluate_v4 --partition validation
python -m training.train_v4
python -m evaluation.evaluate_v4_models --partition validation
python -m evaluation.evaluate_v4_models --partition test
python -m evaluation.evaluate_v4_models --partition default
```

## V5: แผนที่หลากหลายและกรองทิศระหว่างฝึก

V5 สร้างแผนที่ใหม่ 100 แผนที่สำหรับฝึกและ 10 แผนที่สำหรับ validation โดยกันอีก 20 แผนที่ไว้ทดสอบครั้งสุดท้ายใน V6 ทดลองฝึกต่อจาก V4 แบบไม่มีความจำด้วยงบเท่ากันสองชุด: PPO ปกติและ MaskablePPO ที่เลือกเฉพาะทิศที่เดินได้ตั้งแต่ระหว่างฝึก ดู [ตัวอย่างแผนที่](results/v5_map_examples.png)

บน 10 แผนที่ validation แบบกรองทิศจับ Hider V3 ได้ **948/3000 (31.6%)** เทียบกับ Control **993/3000 (33.1%)** และ Seeker สุ่ม **956/3000 (31.9%)** แม้การกรองจะหยุดการเดินชนกำแพงระหว่างฝึก แต่ยังไม่เพิ่มอัตราจับตามเกณฑ์ที่กำหนด และบนแผนที่เดิม V5 ทั้งสองแบบต่ำกว่า V3.1 มาก จึงเก็บ V5 เป็นผลทดลอง ไม่แทนโมเดลเดิม

รายละเอียดผลราย seed รายแผนที่ ข้อจำกัด และ [กราฟ validation](results/v5_validation_metrics.png) อยู่ใน [รายงาน V5](docs/v5_report.md) ทำซ้ำได้ด้วย:

```bash
python -m training.train_v5
python -m evaluation.evaluate_v5 --partition validation
python -m evaluation.evaluate_v5 --partition default
python -m evaluation.evaluate_v5 --partition legacy
```

## V6: หลักสูตรแผนที่และคู่แข่งหลายตัว

V6 เปรียบเทียบ PPO สองแบบด้วยงบฝึกเท่ากัน: ใช้แผนที่ฝึกทั้งหมดตั้งแต่ต้น (`flat`) กับค่อยเพิ่มจาก 28 → 58 → 108 แผนที่ (`curriculum`) ทั้งสองแบบเลือก Hider สุ่มหรือ Hider V3 หนึ่งตัวต่อเกม และฝึก 3 seed ผล validation กับเกณฑ์ถูกล็อกก่อนเปิดแผนที่ทดสอบสุดท้าย V5F

ผล V5F 20 แผนที่: `curriculum` จับได้ **1,257/6,000 (20.9%)**, `flat` **1,120/6,000 (18.7%)**, เทียบกับ Seeker สุ่ม **1,814/6,000 (30.2%)** และ V5 Control **1,815/6,000 (30.2%)** ผลบนแผนที่เดิมของ V6 ก็ลดลงมาก จึง **ไม่มีโมเดล V6 ที่ผ่านเกณฑ์เลื่อนใช้** V5F ถูกเปิดแล้วและไม่ควรใช้เป็นชุดที่ไม่เคยเห็นสำหรับการทดลองครั้งต่อไป

ดู [รายงาน V6](docs/v6_report.md), [แผนที่ล็อกก่อนทดลอง](docs/v6_protocol.md) และ [กราฟ V5F](results/v6_final_metrics.png) รีโปมีโมเดลและผลรายเกมครบแล้ว คำสั่งประเมินซ้ำโดยเก็บผลในโฟลเดอร์ใหม่อยู่ท้ายรายงาน
