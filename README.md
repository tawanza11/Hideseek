# Hide-and-Seek RL V1

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

ข้อมูลเพิ่มเติมสำหรับรายงานอยู่ใน [docs/report.md](docs/report.md) ระดับถัดไปจึงค่อยเพิ่ม Hider ที่เรียนรู้เองและสิ่งกีดขวาง
