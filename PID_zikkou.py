target = 10.0
current = 0.0
Kp = 2.0
Ki = 0.4
Kd = 0.4
drag = 2.0

pid = PID(Kp=Kp, Ki=Ki, Kd=Kd)  

history = []
for i in range(N_steps):
    output = pid.compute(target, current, dt)
    current += (output - drag) * dt
    history.append(current)

plt.plot(history, label="current")
plt.axhline(target, color="gray", linestyle="--", label="target")
plt.text(
    0.98, 0.05,          # ① どこに書くか
    f"Kp={Kp}, Ki={Ki}, Kd={Kd}",   # ② 何を書くか
    transform=plt.gca().transAxes,   # ③ 座標の基準
    ha="right",           # ④ 文字の揃え方
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)  # ⑤ 枠
)
plt.legend()
plt.grid(True)
plt.show()