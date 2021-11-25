import numpy as np
from matplotlib import pyplot as plt
import random

R = 1e-6
D_T = 1 * 1e-13
D_R = 1
v = 3e-6
delta_t = 0.01
variance = 1
parameters = np.array([R, D_T, D_R, v, delta_t, variance])
n_particles = 100
n_timesteps = 100
box_size = 1e-4

position_list = np.zeros([n_particles, 2])
phi_list = np.zeros([n_particles])


def get_distances(_position_candidate, _position_list):
    _distances = np.linalg.norm(_position_candidate - _position_list, axis=1)
    return _distances


def initialize_positions(_position_list, _n_particles, _box_size):
    for i in range(_n_particles):
        _position_list[i, :] = np.random.rand(2) * _box_size
    return _position_list


def initialize_angles(_phi_list, _n_particles):
    for i in range(_n_particles):
        _phi_list[i] = np.random.rand() * 2 * np.pi
    return _phi_list


def simulate_motion(_position, _phi, _delta_t, _D_T, _D_R, _v, _variance, _n_timesteps):
    _new_position = np.zeros(2)
    _W_phi = np.random.normal(loc=0, scale=variance)
    _W_x = np.random.normal(loc=0, scale=variance)
    _W_y = np.random.normal(loc=0, scale=variance)
    _angle = _phi + _W_phi * np.sqrt(2 * _D_R) * np.sqrt(_delta_t)

    _new_position[0] = _position[0] + _v * np.cos(_angle) * _delta_t + np.sqrt(2 * _D_T) * \
                          _W_x * np.sqrt(_delta_t)
    _new_position[1] = _position[1] + _v * np.sin(_angle) * _delta_t + np.sqrt(2 * _D_T) * \
                          _W_y * np.sqrt(_delta_t)
    return np.array(_new_position)


def hard_sphere_correction(_position1, _position2, _R):
    _center_distance = np.linalg.norm(_position1 - _position2)
    _overlap = 2 * _R - _center_distance
    if (_overlap > 0) and (_overlap != 2 * R):
        _distance_to_move = _overlap / 2
        _direction_to_move = (_position1 - _position2) / _center_distance
        _new_position1 = _position1 + _direction_to_move * _distance_to_move
        _new_position2 = _position2 - _direction_to_move * _distance_to_move
        return _new_position1, _new_position2
    elif _overlap == 2 * R:
        _angle = np.random.rand() * 2 * np.pi
        _new_position1 = _position1 + R * np.array([np.cos(_angle), np.sin(_angle)])
        _new_position2 = _position2 - R * np.array([np.cos(_angle), np.sin(_angle)])
        return _new_position1, _new_position2
    else:
        return _position1, _position2


def check_overlap(_position_list, _R, _n_particles):
    for i in range(_n_particles):
        for k in range(_n_particles):
            if i != k:
                _position1, _position2 = hard_sphere_correction(_position_list[i, :], _position_list[k, :], R)
                _position_list[i, :] = _position1
                _position_list[k, :] = _position2
    return _position_list


def outside_box(_position_list, _n_particles, _box_size):
    for i in range(_n_particles):
        if _position_list[i, 0] > _box_size:
            _position_list[i, 0] = _position_list[i, 0] - _box_size
        elif _position_list[i, 0] < 0:
            _position_list[i, 0] = _position_list[i, 0] + _box_size
        if _position_list[i, 1] > _box_size:
            _position_list[i, 1] = _position_list[i, 1] - _box_size
        elif _position_list[i, 1] < 0:
            _position_list[i, 1] = _position_list[i, 1] + _box_size
    return _position_list


def evolve_in_time(_position_list, _phi_list, _n_particles, _box_size, _n_timesteps, _parameters):
    _R, _D_T, _D_R, _v, _delta_t, _variance = parameters
    _position_array = np.zeros([_n_particles, 2, _n_timesteps + 1])
    _position_array[:, :, 0] = _position_list
    for j in range(_n_timesteps):
        print(f'Current timestep: {j}')
        for i in range(_n_particles):
            _position_list[i, :] = simulate_motion(_position_list[i, :], _phi_list[i], _delta_t, _D_T, _D_R, _v,
                                                   _variance, _n_timesteps)
            _position_list = outside_box(_position_list, _n_particles, _box_size)
            _position_list = check_overlap(_position_list, _R, _n_particles)
            _position_array[:, :, j + 1] = _position_list
    return _position_array


position_list = initialize_positions(position_list, n_particles, box_size)
phi_list = initialize_angles(phi_list, n_particles)
position_list = check_overlap(position_list, R, n_particles)
position_array = evolve_in_time(position_list, phi_list, n_particles, box_size, n_timesteps, parameters)

fig1, ax1 = plt.subplots(1, figsize=(8, 8))
disc_vector = np.linspace(0, 2 * np.pi, 100)
position_array *= 1e6    # scale for micrometer
box_size *= 1e6
R *= 1e6
print(position_array)
# color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#              for i in range(n_particles)]
#
# for i in range(n_particles):
#     rgb = np.random.randint(0, 255, 3)
#     ax1.plot(position_array[i, 0, -1] + R * np.cos(disc_vector), position_array[i, 1, -1] + R * np.sin(disc_vector),
#              c=color[i])
#     ax1.plot(position_array[i, 0, 0] + R * np.cos(disc_vector), position_array[i, 1, 0] + R * np.sin(disc_vector),
#              c=color[i], linestyle='dashed', alpha=1)

for i in range(n_particles):
    rgb = np.random.randint(0, 255, 3)
    ax1.plot(position_array[i, 0, -1] + R * np.cos(disc_vector), position_array[i, 1, -1] + R * np.sin(disc_vector),
             c='k')
    ax1.plot(position_array[i, 0, 0] + R * np.cos(disc_vector), position_array[i, 1, 0] + R * np.sin(disc_vector),
             c='k', alpha=0.4)
    ax1.plot(position_array[i, 0, :], position_array[i, 1, :], 'k', alpha=0.8)
ax1.set_aspect('equal')
ax1.set_xlabel('x [$\\mu$m]')
ax1.set_xlim([0, box_size])
ax1.set_ylabel('y [$\\mu$m]')
ax1.set_ylim([0, box_size])
plt.tight_layout()

plt.show()
