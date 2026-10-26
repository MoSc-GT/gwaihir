import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Gitterpunkte je Stufe
N = 100

# Kerbin
mu = 3.5316e12           # [m^3/s^2]
R = 600e3                # [m]
omega = 72.7e-6          # [rad/s]
g0 = 9.80665             # [m/s^2]
cDA = 3              # [m²]

# Zielorbit
h_target = 80e3
r_target = R + h_target
v_circular = np.sqrt(mu / r_target)

# Beispielkonfiguration: zweistufige Rakete
# Alle Massen sind die Masse der jeweiligen Stufe ohne die darüberliegenden Stufen.
# Stufe 1 trägt Stufe 2 und Nutzlast.
stages = [
    {
        "name": "Stufe 1",
        "thrust_N": 25e5,
        "isp_s": 250.0,
        "dry_mass_kg": 4e4,
        "prop_mass_kg": 16e4,
    }
    # {
    #     "name": "Stufe 2",
    #     "thrust_N": 2.5e5,
    #     "isp_s": 350.0,
    #     "dry_mass_kg": 1e4,
    #     "prop_mass_kg": 3e4,
    # },
    # {
    #     "name": "Stufe 3",
    #     "thrust_N": 0.5e4,
    #     "isp_s": 600.0,
    #     "dry_mass_kg": 1e3,
    #     "prop_mass_kg": 3e3,
    # }
]
payload_kg = 4.7e1

# Masse zu Beginn muss alle Stufen und die Nutzlast enthalten
m0 = payload_kg + sum(
    stage["dry_mass_kg"] + stage["prop_mass_kg"]
    for stage in stages
)

# Anfang am Äquator; Geschwindigkeit durch Erdrotation nach Osten
vr0 = 0.0
vt0 = omega * R

opti = ca.Opti()

# Eine eigene Zustandsmatrix und Steuerung für jede Stufe.
# Zustände: r, theta, vr, vt, m
T = []
X = []
alpha = []

for i, stage in enumerate(stages):
    T.append(opti.variable())
    X.append(opti.variable(5, N + 1))
    alpha.append(opti.variable(1, N))

    mdot = stage["thrust_N"] / (stage["isp_s"] * g0)
    max_burn_time = stage["prop_mass_kg"] / mdot

    # Brenndauer begrenzen: Die Stufe darf nicht mehr Treibstoff verbrauchen,
    # als sie enthält.
    opti.subject_to(opti.bounded(1.0, T[i], max_burn_time))

    # Schubwinkel relativ zur lokalen Tangente
    opti.subject_to(
        opti.bounded(-np.pi / 2, alpha[i], np.pi / 2)
    )

    r = X[i][0, :]
    theta = X[i][1, :]
    vr = X[i][2, :]
    vt = X[i][3, :]
    mass = X[i][4, :]

    dt = T[i] / N

    # Radien und Massen müssen physikalisch sinnvoll bleiben
    opti.subject_to(r >= R)
    opti.subject_to(mass >= payload_kg)

    # Anfangszustand der ersten Stufe oder Zustand nach Abwurf der vorherigen
    if i == 0:
        opti.subject_to(r[0] == R)
        opti.subject_to(theta[0] == 0)
        opti.subject_to(vr[0] == vr0)
        opti.subject_to(vt[0] == vt0)
        opti.subject_to(mass[0] == m0)
    else:
        previous = X[i - 1]
        previous_stage = stages[i - 1]

        # Beim Stufenabwurf ändern sich Position und Geschwindigkeit nicht.
        opti.subject_to(r[0] == previous[0, -1])
        opti.subject_to(theta[0] == previous[1, -1])
        opti.subject_to(vr[0] == previous[2, -1])
        opti.subject_to(vt[0] == previous[3, -1])

        # Die leere Trockenmasse der vorherigen Stufe wird abgeworfen.
        opti.subject_to(
            mass[0] == previous[4, -1] - previous_stage["dry_mass_kg"]
        )

    # Dynamik innerhalb der Stufe
    for k in range(N):
        r_k = X[i][0, k]
        theta_k = X[i][1, k]
        vr_k = X[i][2, k]
        vt_k = X[i][3, k]
        m_k = X[i][4, k]

        r_next = X[i][0, k + 1]
        theta_next = X[i][1, k + 1]
        vr_next = X[i][2, k + 1]
        vt_next = X[i][3, k + 1]
        m_next = X[i][4, k + 1]

        thrust = stage["thrust_N"]
        isp = stage["isp_s"]
        mdot = thrust / (isp * g0)
        angle = alpha[i][0, k]

        # Drag Berechnung
        H = (r_k + r_next) / 2 - R  # Höhe über der Oberfläche, ungefähr am Intervallmittelpunkt

        p = 1013.25e2 * ca.exp(
            -9.81 * 0.0289644 * H / (8.3143 * 293)
        )
        rho = p / (287.058 * 293)
        v_rel_earth = np.sqrt(vr_k**2 + (vt_k - vt0)**2)
        F_drag = ca.if_else(
            H < 75e3,
            0.5 * rho * cDA * (v_rel_earth**2 / 4),
            0
        )
        dir_v_rel_earth = np.arctan2(vr_k+1e-9, (vt_k - vt0))

        # Bewegungsgleichungen in Polarkoordinaten
        ar_k = vt_k**2 / r_k - mu / r_k**2 + (thrust / m_k) * ca.sin(angle) - (F_drag/m_k)*ca.sin(dir_v_rel_earth)
        at_k = -vr_k * vt_k / r_k + (thrust / m_k) * ca.cos(angle) - (F_drag/m_k)*ca.cos(dir_v_rel_earth)

        ar_next = (
            vt_next**2 / r_next
            - mu / r_next**2
            + (thrust / m_next) * ca.sin(angle)  - (F_drag/m_k)*ca.sin(dir_v_rel_earth)
        )
        at_next = (
            -vr_next * vt_next / r_next
            + (thrust / m_next) * ca.cos(angle)  - (F_drag/m_k)*ca.cos(dir_v_rel_earth)
        )

        # Trapez-Kollokation
        opti.subject_to(
            r_next - r_k == dt * (vr_k + vr_next) / 2
        )
        opti.subject_to(
            theta_next - theta_k
            == dt * (vt_k / r_k + vt_next / r_next) / 2
        )
        opti.subject_to(
            vr_next - vr_k == dt * (ar_k + ar_next) / 2
        )
        opti.subject_to(
            vt_next - vt_k == dt * (at_k + at_next) / 2
        )
        opti.subject_to(m_next - m_k == -dt * mdot)

    # Nur Stufen, die vor der nächsten Stufe abgeworfen werden,
    # müssen ihren Treibstoff vollständig verbraucht haben.
    # if i < len(stages) - 1:
    #     opti.subject_to(
    #         X[i][4, -1]
    #         == X[i][4, 0] - stages[i]["prop_mass_kg"]
    #     )

# Zielorbit nach Brennschluss der letzten Stufe
last = X[-1]
opti.subject_to(last[0, -1] == r_target)
opti.subject_to(last[2, -1] == 0)
opti.subject_to(last[3, -1] == v_circular)

# print(T[1])
# Gesamtbrenndauer minimieren
opti.minimize(T[len(stages)-1])


# Startwerte
for i in range(len(stages)):
    stage = stages[i]
    mdot = stage["thrust_N"] / (stage["isp_s"] * g0)
    opti.set_initial(T[i], 0.7 * stage["prop_mass_kg"] / mdot)
    opti.set_initial(alpha[i], 0.1)
# Sinnvolle Startwerte für IPOPT.
# Ohne diese Werte startet CasADi Zustände wie r und m zunächst bei null,
# obwohl sie in den Bewegungsgleichungen als Nenner auftreten.
mass_at_stage_start = m0

for i, stage in enumerate(stages):
    mdot_i = stage["thrust_N"] / (stage["isp_s"] * g0)

    # Die Endmassenbedingung legt die Brenndauer bereits fest:
    # prop_mass = mdot * T
    burn_guess = stage["prop_mass_kg"] / mdot_i
    opti.set_initial(T[i], burn_guess)

    if i > 0:
        mass_at_stage_start -= stages[i - 1]["prop_mass_kg"]
        mass_at_stage_start -= stages[i - 1]["dry_mass_kg"]

    mass_at_stage_end = mass_at_stage_start - stage["prop_mass_kg"]

    for k in range(N + 1):
        f = k / N

        # Positive, glatte Anfangsbahn
        opti.set_initial(
            X[i][0, k],
            R + f * h_target * (i + 1) / len(stages)
        )
        opti.set_initial(X[i][1, k], 0.5 * f * (i + 1))
        opti.set_initial(X[i][2, k], 100.0 * np.sin(np.pi * f))
        opti.set_initial(
            X[i][3, k],
            vt0 + f * (v_circular - vt0)
        )
        opti.set_initial(
            X[i][4, k],
            mass_at_stage_start + f * (mass_at_stage_end - mass_at_stage_start)
        )

    opti.set_initial(alpha[i], 0.1)
# IPOPT
opti.solver(
    "ipopt",
    {"expand": True},
    {
        "print_level": 5,
        "max_iter": 4000,
        "tol": 1e-7,
    },
)

solution = opti.solve()

# Ergebnisse auslesen und Zeitabschnitte zusammensetzen
all_time = []
all_r = []
all_vr = []
all_vt = []
all_mass = []
all_alpha = []

time_offset = 0.0

for i, stage in enumerate(stages):
    Ti = float(solution.value(T[i]))
    Xi = np.array(solution.value(X[i]))
    alphai = np.array(solution.value(alpha[i])).flatten()

    stage_time = np.linspace(time_offset, time_offset + Ti, N + 1)

    # Den gemeinsamen Trennpunkt nur einmal plotten
    if i > 0:
        stage_time = stage_time[1:]
        Xi = Xi[:, 1:]

    all_time.extend(stage_time)
    all_r.extend(Xi[0, :])
    all_vr.extend(Xi[2, :])
    all_vt.extend(Xi[3, :])
    all_mass.extend(Xi[4, :])
    all_alpha.extend(alphai)

    print(f"{stage['name']}: Brenndauer {Ti:.2f} s")
    print(
        f"  Masse vor Trennung: "
        f"{float(solution.value(X[i][4, -1])):.1f} kg"
    )

    time_offset += Ti

all_time = np.array(all_time)
all_r = np.array(all_r)
all_vr = np.array(all_vr)
all_vt = np.array(all_vt)
all_mass = np.array(all_mass)

print(f"Gesamtbrenndauer: {time_offset:.2f} s")
print(f"Nutzlastmasse: {payload_kg:.1f} kg")
print(f"Zielhöhe: {(all_r[-1] - R) / 1000:.2f} km")
print(f"Endgeschwindigkeit tangential: {all_vt[-1]:.2f} m/s")
print(f"Kreisbahngeschwindigkeit: {v_circular:.2f} m/s")

# Plots
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

axes[0].plot(all_time, (all_r - R) / 1000)
axes[0].set_ylabel("Höhe [km]")
axes[0].grid(True)

axes[1].plot(all_time, all_vr, label="radial")
axes[1].plot(all_time, all_vt, label="tangential")
axes[1].set_ylabel("Geschwindigkeit [m/s]")
axes[1].legend()
axes[1].grid(True)


azimuth_deg = np.rad2deg(
    np.arctan2(
        (all_vr[:-1]),
        (all_vt[:-1])
    )
)
aoa_deg = np.rad2deg(all_alpha) - azimuth_deg

time_mid = (all_time[:-1] + all_time[1:]) / 2

axes[2].plot(all_time[:-1], np.rad2deg(all_alpha), label="Pitchwinkel")
axes[2].plot(all_time[:-1], azimuth_deg, label="Flugazimuth")
axes[2].plot(all_time[:-1], aoa_deg, label="Anstellwinkel")

axes[2].set_ylabel("Winkel [°]")
axes[2].set_ylim(-90, 90)
axes[2].set_yticks(np.arange(-90, 91, 30))
axes[2].legend()
axes[2].grid(True)

# Treibstoffinhalt jeder Stufe in Prozent über der Gesamtzeit
axes[3].set_ylabel("Treibstoff [%]")
axes[3].set_xlabel("Zeit [s]")
axes[3].set_ylim(0, 105)

time_offset = 0.0
for i, stage in enumerate(stages):
    Ti = float(solution.value(T[i]))
    Xi = np.array(solution.value(X[i]))

    # Treibstoffverbrauch der Stufe aus der Raketenmasse ableiten.
    # Die Trockenmasse dieser Stufe bleibt während ihres Brennens an Bord.
    prop_remaining = (
        Xi[4, :]
        - payload_kg
        - sum(s["dry_mass_kg"] for s in stages[i:])
        - sum(s["prop_mass_kg"] for s in stages[i + 1:])
    )
    prop_percent = 100 * prop_remaining / stage["prop_mass_kg"]

    # Vor Zündung: 100 %. Während des Brennens: berechneter Verlauf.
    stage_time = np.linspace(time_offset, time_offset + Ti, N + 1)
    plot_time = np.concatenate(([0.0] if i == 0 else [time_offset], stage_time))
    plot_percent = np.concatenate(([100.0] if i == 0 else [100.0], prop_percent))

    # Gemeinsamer Gesamtzeitvektor für Stufen, die noch nicht gezündet haben:
    if i == 0:
        axes[3].plot(plot_time, plot_percent, label=stage["name"])
    else:
        axes[3].plot(plot_time, plot_percent, label=stage["name"])

    time_offset += Ti

axes[3].legend()
axes[3].grid(True)

fig.suptitle(f"Aufstieg, Gesamtbrenndauer = {time_offset:.1f} s")
fig.tight_layout()
plt.show()