import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Gitterpunkte je Stufe
N = 30

# Kerbin
mu = 3.5316e12           # [m^3/s^2]
mu = 4e14                # [m^3/s^2]
R = 6730e3                # [m]
omega = 72.7e-6          # [rad/s]
g0 = 9.80665             # [m/s^2]
cDA = 1               # Cd*A [m^2]

# Einfaches isothermes Atmosphaerenmodell
p0 = 101325.0            # [Pa]
T_atm = 293.0            # [K]
M_air = 0.0289644        # [kg/mol]
R_gas = 8.3143           # [J/(mol K)]
R_air = 287.058          # [J/(kg K)]
atmosphere_cutoff = 75e3 # [m]

# Startposition (Ostlaenge positiv) und Zielorbit
start_latitude_deg = 0.0
start_longitude_deg = 2.0
start_altitude = 0.0     # [m ueber R]
h_target = 300e3         # [m]
inclination_target_deg = 0
if not 0 <= inclination_target_deg <= 90:
    raise ValueError("Dieses Beispiel unterstuetzt prograde Inklinationen von 0 bis 90 Grad.")

r_target = R + h_target
v_circular = np.sqrt(mu / r_target)
inc_target = np.deg2rad(inclination_target_deg)

# Stufen: alle Stufen vor der letzten werden leergebrannt und abgeworfen.
# stages = [
#     {"name": "Stufe 1", "thrust_N": 50e5, "isp_s": 280.0,
#      "dry_mass_kg": 4e4, "prop_mass_kg": 16e4},
#     {"name": "Stufe 2", "thrust_N": 4e5, "isp_s": 330.0,
#      "dry_mass_kg": 1e4, "prop_mass_kg": 3e4}
#     # {"name": "Stufe 3", "thrust_N": 0.5e5, "isp_s": 400.0,
#     #  "dry_mass_kg": 1e3, "prop_mass_kg": 3e3},
# ]
stages = [
    {"name": "Stufe 1", "thrust_N": 45e6, "isp_s": 280.0,
     "dry_mass_kg": 0.2e6, "prop_mass_kg": 1.8e6},
    {"name": "Stufe 2", "thrust_N": 5e6, "isp_s": 380.0,
     "dry_mass_kg": 0.05e6, "prop_mass_kg": 0.5e6}
    # {"name": "Stufe 3", "thrust_N": 0.5e5, "isp_s": 400.0,
    #  "dry_mass_kg": 1e3, "prop_mass_kg": 3e3},
]
payload_kg = 6e4



lat0 = np.deg2rad(start_latitude_deg)
lon0 = np.deg2rad(start_longitude_deg)
r0_mag = R + start_altitude
r0 = r0_mag * np.array([np.cos(lat0)*np.cos(lon0), np.cos(lat0)*np.sin(lon0), np.sin(lat0)])
omega_vec = np.array([0.0, 0.0, omega])
v0 = np.cross(omega_vec, r0)  # Anfangsgeschwindigkeit durch Erdrotation


def unit(v):
    return v / ca.sqrt(ca.dot(v, v))

def nedFromDirection(r):
    """Lokale Einheitsvektoren Up, East und North (Singularitaet genau am Pol)."""
    down = -unit(r)
    east = unit(ca.cross(down, ca.DM([0, 0, 1])))
    north = ca.cross(east, down)
    return down, east, north

def density_at(h):
    rho = (p0 * ca.exp(-9.81*M_air*h/(R_gas*T_atm))) / (R_air*T_atm)
    return ca.if_else(h < atmosphere_cutoff, rho, 0.0)

def acceleration(r, v, m, pitch, heading, thrust):
    """Pitch: 0 horizontal, +90 radial aufwaerts; Heading: 0 Nord, +90 Ost."""
    radius = ca.sqrt(ca.dot(r, r))
    down, east, north = nedFromDirection(r)
    gravity = -mu*r/(radius**3)
    acc_down = down*(thrust/m)*ca.sin(-pitch)      # acceleration radial inward
    acc_east = east*(thrust/m)*ca.cos(pitch)*ca.sin(heading)         #acceleration to east
    acc_north = north*(thrust/m)*ca.cos(pitch)*ca.cos(heading)         #acceleration to east
    acc_thrust = acc_north + acc_east + acc_down
    
    # Atmosphaere rotiert mit dem Planeten. Drag wirkt gegen den vollen
    # Geschwindigkeitsvektor relativ zur Luft, also radial und tangential.
    v_air = ca.cross(ca.DM(omega_vec), r)
    v_rel = v - v_air
    v_rel_norm = ca.sqrt(ca.dot(v_rel, v_rel) + 1e-12)
    drag_acc = -0.5*density_at(radius-R)*cDA*v_rel_norm*v_rel/m

    return acc_thrust + gravity + drag_acc


# Zustaende je Gitterpunkt: x,y,z,vx,vy,vz,m
# Steuerungen je Intervall: Pitch und Heading
opti = ca.Opti()
T, X, pitch, heading, payload_opt = [], [], [], [], []
payload_opt.append(opti.variable(1, 1))
m0 = payload_opt[0] + sum(s["dry_mass_kg"] + s["prop_mass_kg"] for s in stages)
for i, stage in enumerate(stages):
    T.append(opti.variable())
    X.append(opti.variable(7, N+1))
    pitch.append(opti.variable(1, N))
    heading.append(opti.variable(1, N))
    mdot = stage["thrust_N"]/(stage["isp_s"]*g0)
    opti.subject_to(opti.bounded(1.0, T[i], stage["prop_mass_kg"]/mdot))
    opti.subject_to(opti.bounded(-np.pi/2, pitch[i], np.pi/2))
    opti.subject_to(opti.bounded(-0.1, heading[i], np.pi))  
    opti.subject_to(opti.bounded(1, payload_opt[0], 1e3))


    r = X[i][0:3, :]
    v = X[i][3:6, :]
    m = X[i][6, :]
    for k in range(N + 1):
        opti.subject_to(ca.dot(r[:, k], r[:, k]) / R**2 >= 1)
    opti.subject_to(m >= [0])

    if i == 0:
        opti.subject_to(r[:, 0] == r0)
        opti.subject_to(v[:, 0] == v0)
        opti.subject_to(m[0] == m0)
    else:
        prev = X[i-1]
        opti.subject_to(r[:, 0] == prev[0:3, -1])
        opti.subject_to(v[:, 0] == prev[3:6, -1])
        # Vorstufe ist leer; ihre Trockenmasse wird abgeworfen.
        opti.subject_to(m[0] == prev[6, -1] - stages[i-1]["dry_mass_kg"])

    dt = T[i]/N
    for k in range(N):
        rk, vk, mk = r[:, k], v[:, k], m[k]
        rn, vn, mn = r[:, k+1], v[:, k+1], m[k+1]
        a0 = acceleration(rk, vk, mk, pitch[i][0, k], heading[i][0, k], stage["thrust_N"])
        a1 = acceleration(rn, vn, mn, pitch[i][0, k], heading[i][0, k], stage["thrust_N"])
        opti.subject_to((rn - rk - dt*(vk + vn)/2) / R == 0)
        opti.subject_to((vn - vk - dt*(a0 + a1)/2) / v_circular == 0)
        opti.subject_to((mn - mk + dt*stage["thrust_N"]/(stage["isp_s"]*g0)) / m0 == 0)
    # Stufen, vor denen eine weitere Stufe gezuendet wird, muessen leer sein.
    if i < len(stages)-1:
        opti.subject_to(X[i][6, -1] == X[i][6, 0] - stage["prop_mass_kg"])

# Kreisbahn in Zielhoehe und Zielinklination
last = X[-1]
r_f, v_f = last[0:3, -1], last[3:6, -1]
r_f_norm = ca.sqrt(ca.dot(r_f, r_f))
h_vec = ca.cross(r_f, v_f)
h_norm = ca.sqrt(ca.dot(h_vec, h_vec) + 1e-12)
h_scale = r_target * v_circular

opti.subject_to(r_f_norm / r_target == 1)
opti.subject_to(ca.dot(r_f, v_f) / h_scale == 0)
opti.subject_to(ca.dot(v_f, v_f) / v_circular**2 == 1)

if abs(inc_target) < 1e-10:
    # Äquatoriale Bahn: Drehimpuls zeigt entlang der Z-Achse
    opti.subject_to(h_vec[0] / h_scale == 0)
    opti.subject_to(h_vec[1] / h_scale == 0)
    opti.subject_to(h_vec[2] >= 0)
else:
    opti.subject_to(h_vec[2] / h_norm == np.cos(inc_target))
    opti.subject_to(h_vec[2] >= 0)

# opti.subject_to((payload_kg - payload_opt[0]) / (payload_kg+1e-12) == 0)

# Wie in der aktuellen Fassung: Brenndauer der letzten Stufe minimieren.
# opti.minimize(T[len(stages)-1])
opti.minimize(-payload_opt[0])

# Startwerte
mass_start = payload_kg + sum(s["dry_mass_kg"] + s["prop_mass_kg"] for s in stages)
east0 = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
north0 = np.cross(r0/r0_mag, east0)
heading_guess = np.deg2rad(90.0)
tangent0 = np.cos(heading_guess)*north0 + np.sin(heading_guess)*east0
for i, stage in enumerate(stages):
    mdot = stage["thrust_N"]/(stage["isp_s"]*g0)
    burn_guess = 0.8*stage["prop_mass_kg"]/mdot
    opti.set_initial(T[i], burn_guess)
    if i > 0:
        mass_start -= stages[i-1]["prop_mass_kg"] + stages[i-1]["dry_mass_kg"]
    mass_end = mass_start - mdot*burn_guess
    for k in range(N+1):
        f = k/N
        rg = r0 + f*h_target*(0.25*tangent0 + 0.97*r0/r0_mag)
        vg = v0 + f*(v_circular*tangent0-v0)
        opti.set_initial(X[i][0:3, k], rg)
        opti.set_initial(X[i][3:6, k], vg)
        opti.set_initial(X[i][6, k], mass_start + f*(mass_end-mass_start))
    opti.set_initial(pitch[i], np.linspace(np.deg2rad(80), 0, N))
    opti.set_initial(heading[i], heading_guess)
opti.set_initial(payload_opt[0], 1e-3)

opti.solver("ipopt", {"expand": True},
            {"print_level": 5, "max_iter": 4000, "tol": 1e-7})
sol = opti.solve()

# Ergebnisse auslesen
all_time, all_r, all_v, all_mass = [], [], [], []
all_pitch, all_heading = [], []
stage_starts = [0.0]
time_offset = 0.0
for i, stage in enumerate(stages):
    Ti = float(sol.value(T[i]))
    Xi = np.array(sol.value(X[i]))
    ti = np.linspace(time_offset, time_offset+Ti, N+1)
    if i > 0:
        ti, Xi = ti[1:], Xi[:, 1:]
    all_time.extend(ti)
    all_r.extend(Xi[0:3, :].T)
    all_v.extend(Xi[3:6, :].T)
    all_mass.extend(Xi[6, :])
    all_pitch.extend(np.array(sol.value(pitch[i])).flatten())
    all_heading.extend(np.array(sol.value(heading[i])).flatten())
    print(f"{stage['name']}: Brenndauer {Ti:.2f} s")
    print(f"  Masse vor Trennung/Brennschluss: {float(sol.value(X[i][6, -1])):.1f} kg")
    time_offset += Ti
    stage_starts.append(time_offset)

all_time = np.asarray(all_time)
all_r = np.asarray(all_r)
all_v = np.asarray(all_v)
all_mass = np.asarray(all_mass)
all_pitch = np.asarray(all_pitch)
all_heading = np.asarray(all_heading)
radius = np.linalg.norm(all_r, axis=1)
altitude = radius-R
speed = np.linalg.norm(all_v, axis=1)
radial_speed = np.sum(all_r*all_v, axis=1)/radius
v_air = np.cross(np.broadcast_to(omega_vec, all_r.shape), all_r)
v_rel = all_v-v_air
air_relative_speed = np.linalg.norm(v_rel, axis=1)
h_final = np.cross(all_r[-1], all_v[-1])
inc_final = np.rad2deg(np.arccos(np.clip(h_final[2]/np.linalg.norm(h_final), -1, 1)))
print(f"Gesamtbrenndauer: {time_offset:.2f} s")
print(f"Zielhoehe: {altitude[-1]/1000:.2f} km")
print(f"Endgeschwindigkeit inertial: {speed[-1]:.2f} m/s")
print(f"Erreichte Inklination: {inc_final:.2f} deg")
payload_value = float(sol.value(payload_opt[0]))
print(f"Optimierte Nutzlast: {payload_value:.1f} kg")


# Plots
fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
axes[0].plot(all_time, altitude/1000)
axes[0].set_ylabel("Hoehe [km]")
axes[0].grid(True)
axes[1].plot(all_time, speed, label="inertial")
axes[1].plot(all_time, air_relative_speed, label="relativ zur Atmosphaere")
axes[1].plot(all_time, radial_speed, label="radial")
axes[1].set_ylabel("Geschwindigkeit [m/s]")
axes[1].legend()
axes[1].grid(True)
time_mid = (all_time[:-1]+all_time[1:])/2
axes[2].plot(time_mid, np.rad2deg(all_pitch), label="Pitch")
axes[2].plot(time_mid, np.rad2deg(all_heading), label="Heading ab Nord")
axes[2].set_ylabel("Winkel [deg]")
axes[2].set_yticks(np.arange(-90, 91, 30))
axes[2].legend()
axes[2].grid(True)

axes[3].set_ylabel("Treibstoff [%]")
axes[3].set_xlabel("Zeit [s]")
axes[3].set_ylim(0, 105)
for i, stage in enumerate(stages):
    Xi = np.array(sol.value(X[i]))
    other_mass = (payload_value
                  + sum(s["dry_mass_kg"] for s in stages[i:])
                  + sum(s["prop_mass_kg"] for s in stages[i+1:]))
    prop_pct = 100*(Xi[6, :]-other_mass)/stage["prop_mass_kg"]
    ti = np.linspace(stage_starts[i], stage_starts[i+1], N+1)
    axes[3].plot(np.r_[0.0, ti, all_time[-1]],
                 np.r_[100.0, prop_pct, prop_pct[-1]], label=stage["name"])
axes[3].legend()
axes[3].set_yticks(np.arange(0, 110, 10))
axes[3].grid(True)
fig.suptitle(f"3D-Aufstieg: h={altitude[-1]/1000:.1f} km, i={inc_final:.1f} deg")
fig.tight_layout()
plt.show()
