from trimaran_vessel import trimaran_vessel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pid import PID
import matplotlib.animation as animation

dt = 0.1               #1ステップに進む秒数
N_steps = 5000    #*0.1秒

boat = trimaran_vessel()
state = boat.get_state()
goal = np.array([50,50])

pid_heading = PID(Kp=15.0, Ki=0, Kd=0.0)
pid_speed = PID(Kp=15.0, Ki=0.1, Kd=1.0)

x_history = []
y_history = []
speed_history = []

for i in range(N_steps):
    state = boat.get_state() 
    current_pos = np.array([state[0],state[1]])
    current_theta = state[2]
    current_speed = state[3]

    vec_to_goal = goal - current_pos
    goal_angle = np.arctan2(vec_to_goal[1],vec_to_goal[0])
    relative_angle = goal_angle - current_theta
    relative_angle = (relative_angle + np.pi) % (2*np.pi) - np.pi

    N = pid_heading.compute(target=0.0, current=-relative_angle, dt=dt)
    max_speed = 0.5
    distance = np.linalg.norm(vec_to_goal)
    target_speed = min(max_speed, distance * 0.5)
    X_out = pid_speed.compute(target=target_speed, current=current_speed, dt=dt)
    u1 = X_out                                    # ⑤ u1をPIDの出力にする
    u2 = u1 + N
    u3 = u1 - N
    u1, u2, u3 = np.clip([u1, u2, u3], -10, 100)

    boat.set_control_input([u1, u2, u3, 0, 0, 0])
    boat.update_state(dt)

    x_history.append(state[0])
    y_history.append(state[1])
    speed_history.append(current_speed)

print(f"記録したステップ数: {len(x_history)}")

# --- ② アニメーションを作る ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(min(x_history)-5, max(x_history)+5)
ax.set_ylim(min(y_history)-5, max(y_history)+5)
ax.plot(goal[0], goal[1], "rx", markersize=12)
ax.plot(0, 0, "go")
ax.grid(True)
ax.set_aspect("equal")

line, = ax.plot([], [], "b-", linewidth=1)
point, = ax.plot([], [], "bo", markersize=8)
text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

skip = 5   # 5ステップに1コマ(間引いて軽くする)

def update(frame):
    idx = frame * skip
    line.set_data(x_history[:idx+1], y_history[:idx+1])
    point.set_data([x_history[idx]], [y_history[idx]])
    text.set_text(f"step={idx}  speed={speed_history[idx]:.2f} m/s")
    return line, point, text

n_frames = len(x_history) // skip
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

# --- ③ 保存(GIFならffmpeg不要) ---
ani.save("goal_pid_movie.gif", writer="pillow", fps=20)
print("保存しました: goal_pid_movie.gif")

plt.show()   # 画面でも確認したい場合