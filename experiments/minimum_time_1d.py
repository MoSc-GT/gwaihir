import casadi as ca
import matplotlib.pyplot as plt


# Anzahl der Zeitschritte im Kollokationsgitter
N = 1000

# Problemparameter
x_start = 15000.0       # Startposition [m]
x_target = 0.0    # Zielposition [m]
v_start = 0.0       # Startgeschwindigkeit [m/s]
v_target = 0.0       # Zielgeschwindigkeit [m/s]
a_max = 1          # Beschleunigungsgrenze [m/s²]

# Optimierungsproblem
opti = ca.Opti()

T = opti.variable()       # Gesamtzeit
X = opti.variable(2, N + 1)  # Zeile 0: Position, Zeile 1: Geschwindigkeit
A = opti.variable(1, N)      # Beschleunigung je Zeitschritt

dt = T / N

# Zielfunktion: minimale Gesamtzeit
opti.minimize(dt * ca.sumsqr(A+9.81))
# opti.minimize(T)

# Randbedingungen
opti.subject_to(T >= 0.01)
opti.subject_to(T < 1000)
# opti.subject_to(X > 0)
opti.subject_to(X[0, 0] == x_start)
opti.subject_to(X[1, 0] == v_start)
opti.subject_to(X[0, N] == x_target)
opti.subject_to(X[1, N] == v_target)

# Beschleunigungsgrenze
opti.subject_to(opti.bounded(-9.81, A, a_max))

# Trapez-Kollokation für:
# Position' = Geschwindigkeit
# Geschwindigkeit' = Beschleunigung
for k in range(N):
    x_k = X[0, k]
    v_k = X[1, k]
    a_k = A[0, k]

    x_next = X[0, k + 1]
    v_next = X[1, k + 1]

    opti.subject_to(x_next - x_k == dt * (v_k + v_next) / 2)
    opti.subject_to(v_next - v_k == dt * a_k)

# IPOPT konfigurieren
opti.solver(
    "ipopt",
    {"expand": True},
    {
        "print_level": 0,
        "max_iter": 2000,
        "tol": 1e-8,
    },
)

# Startwerte für den Solver
opti.set_initial(T, 12.0)
opti.set_initial(X[0, :], [x_start + (x_target - x_start) * k / N for k in range(N + 1)])
opti.set_initial(X[1, :], 0.0)
opti.set_initial(A, 0.0)

# Problem lösen
solution = opti.solve()

# Lösung auslesen
T_opt = float(solution.value(T))
x_opt = solution.value(X[0, :]).flatten()
v_opt = solution.value(X[1, :]).flatten()
a_opt = solution.value(A).flatten()

# Zeitgitter für Zustände und Steuerung
time_states = [k * T_opt / N for k in range(N + 1)]
time_control = [(k + 0.5) * T_opt / N for k in range(N)]

# Für die Darstellung die Beschleunigung an den Mittelpunkten ergänzen
a_plot = list(a_opt) + [a_opt[-1]]
time_acceleration = time_states

# Diagramme
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

axes[0].plot(time_states, x_opt, color="tab:blue")
axes[0].set_ylabel("Weg [m]")
axes[0].grid(True)

axes[1].plot(time_states, v_opt, color="tab:green")
axes[1].set_ylabel("Geschwindigkeit [m/s]")
axes[1].grid(True)

axes[2].step(time_states, a_plot, where="post", color="tab:red")
axes[2].set_ylabel("Beschleunigung [m/s²]")
axes[2].set_xlabel("Zeit [s]")
axes[2].grid(True)

fig.suptitle(f"Zeitoptimale 1D-Bewegung (T = {T_opt:.2f} s)")
fig.tight_layout()
plt.show()