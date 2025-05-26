"""
PyGui implementation of 1-D chain of n particles following the FPUT dynamics
- Live plot of the chain over time
- Live plot of the error over time
"""
import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()
dpg.create_viewport(title='FPUT Simulation', width=1080, height=800)

""" Global flags """
RUNNING = False
FRAME = 0


""" Default values """
n = 10
alpha = 0
beta = 1
dt = 0.1


y_lim = 1
frame_update_rate = 6

x = [i/n for i in range(n+1)]

# Linear string setup
q_linear = np.array([np.sin(i/n*np.pi) for i in range(n+1)])
dot_q_linear = np.array([0.0 for i in range(n+1)])

# Non-linear string setup
q_nonlinear = np.array([np.sin(i/n*np.pi) for i in range(n+1)])
dot_q_nonlinear = np.array([0.0 for i in range(n+1)])

# Energy over time setup
x_time = [i for i in range(100)]
linear_energy = [0 for i in range(100)]
nonlinear_energy = [0 for i in range(100)]
initial_energy = 0

"""
String update functions
"""
# Linear string update function
def update_linear_model():
    global q_linear, dot_q_linear, dt, n
    ddot_q = np.zeros_like(q_linear)
    for j in range(1,n):
        ddot_q[j] = -(2*q_linear[j]-q_linear[j+1]-q_linear[j-1]) 
    dot_q_linear[1:-1] += dt*ddot_q[1:-1]
    q_linear[1:-1] += dt*dot_q_linear[1:-1]
    dpg.set_value("linear fput", [x, q_linear])

# Non-linear string update function
def update_nonlinear_model():
    global q_nonlinear, dot_q_nonlinear, dt, n, alpha, beta
    ddot_q = np.zeros_like(q_nonlinear)
    for j in range(1,n):
        linear_part = -(2*q_nonlinear[j]-q_nonlinear[j+1]-q_nonlinear[j-1])
        non_linear_part_alpha = -(alpha*((q_nonlinear[j]-q_nonlinear[j+1])**2+(q_nonlinear[j]-q_nonlinear[j-1])**2))
        non_linear_part_beta = -(beta*((q_nonlinear[j]-q_nonlinear[j+1])**3+(q_nonlinear[j]-q_nonlinear[j-1])**3))
        ddot_q[j] = linear_part + non_linear_part_alpha + non_linear_part_beta
    dot_q_nonlinear[1:-1] += dt*ddot_q[1:-1]
    q_nonlinear[1:-1] += dt*dot_q_nonlinear[1:-1]
    dpg.set_value("nonlinear fput", [x, q_nonlinear])

def dynamic_y_axis():
    global y_lim
    y_lim = 1.1*max(np.max(np.abs(q_linear)),np.max(np.abs(q_nonlinear)))
    dpg.set_axis_limits("y_axis",-y_lim,y_lim)

"""
Hamiltonian functions
"""
def calculate_fput_H(q,q_dot):
    global alpha, beta
    kinetic = np.sum(q_dot**2)/2
    potential = np.sum(0.5*(q[:-1]-q[1:])**2 + alpha*(q[:-1]-q[1:])**3 + beta*(q[:-1]-q[1:])**4)
    return kinetic + potential

def update_linear_energy():
    global q_linear, dot_q_linear, linear_energy
    energy = calculate_fput_H(q_linear, dot_q_linear)
    linear_energy.append(energy)
    linear_energy.pop(0)
    dpg.set_value("linear fput energy", [x_time, linear_energy])
    
    
def update_non_linear_energy():
    global q_nonlinear, dot_q_nonlinear, nonlinear_energy
    energy = calculate_fput_H(q_nonlinear, dot_q_nonlinear)
    nonlinear_energy.append(energy)
    nonlinear_energy.pop(0)
    dpg.set_value("nonlinear fput energy", [x_time, nonlinear_energy])

def dynamic_energy_axis():
    high_energy = max(initial_energy,np.max(nonlinear_energy),np.max(linear_energy))
    low_energy = min(initial_energy,np.min(nonlinear_energy),np.min(linear_energy))
    dpg.set_axis_limits("energy_axis", low_energy*0.9, high_energy*1.1)

initial_energy = calculate_fput_H(q_linear, dot_q_linear)
linear_energy = [initial_energy for i in range(100)]
nonlinear_energy = [initial_energy for i in range(100)]
"""
pygui code
"""

# Save button functionality
def save_values():
    global n, alpha, beta, dt, x, q_linear, dot_q_linear, q_nonlinear, dot_q_nonlinear, FRAME, frame_update_rate
    global x_time, linear_energy, nonlinear_energy
    n = int(dpg.get_value("slider_n"))
    alpha = dpg.get_value("slider_alpha")
    beta = dpg.get_value("slider_beta")
    dt = dpg.get_value("slider_dt")
    y_lim = dpg.get_value("slider_axes")
    frame_update_rate = dpg.get_value("slider_frame_update")
    dpg.set_axis_limits("y_axis", -y_lim, y_lim)
    
    FRAME = 0
    
    x = [i/n for i in range(n+1)]
    q_linear = np.array([np.sin(i/n*np.pi) for i in range(n+1)])
    dot_q_linear = np.array([0.0 for i in range(n+1)])
    q_nonlinear = np.array([np.sin(i/n*np.pi) for i in range(n+1)])
    dot_q_nonlinear = np.array([0.0 for i in range(n+1)])

    initial_energy = calculate_fput_H(q_linear, dot_q_linear)
    linear_energy = [initial_energy for i in range(100)]
    nonlinear_energy = [initial_energy for i in range(100)]    

    dpg.set_value("params display", f"Current values:\n n={n},\n alpha={alpha},\n beta={beta},\n dt={dt}")
    dpg.set_value("reference energy", [x_time, [initial_energy for i in range(100)]])
    dpg.set_axis_limits("energy_axis", initial_energy*0.7, initial_energy*1.3)

with dpg.window(label="Parameters", pos=(0,0), width=450,height=650):
    dpg.add_text(
        """
        Discrete FPUT model parameters:
        - n: number of nodes
        - alpha: non-linearity coefficient
        - beta: non-linearity coefficient
        - dt: timestep per frame
        
        Other params:
        - axes limits : y-axis limits
        
        Note: Hitting save will reset the current simulation.
        """
    )
    dpg.add_button(label="Save", callback=save_values)
    dpg.add_slider_int(label='n', default_value = n, min_value=3, max_value=100, tag = 'slider_n')
    dpg.add_slider_float(label="Alpha", default_value = alpha, min_value=-0.25, max_value=0.25, tag = 'slider_alpha')
    dpg.add_slider_float(label='Beta', default_value = beta, min_value=0.5, max_value=1.5, tag = 'slider_beta')
    dpg.add_slider_float(label="dt", default_value=dt, min_value=0.01, max_value=0.5, tag = 'slider_dt')
    dpg.add_slider_float(label='axes limits', default_value=y_lim, min_value=1, max_value=10, tag = 'slider_axes')
    dpg.add_slider_int(label='frame update rate', default_value = frame_update_rate, min_value=1, max_value=10, tag = 'slider_frame_update')
    dpg.add_text(f"Current values:\n n={n},\n alpha={alpha},\n beta={beta},\n dt={dt},\n axes limits={y_lim},\n frame update rate={frame_update_rate}", tag="params display")

# Play button functionality
def start_simulation():
    global RUNNING
    RUNNING = True

# Pause button functionality
def stop_simulation():
    global RUNNING
    RUNNING = False

# Checkbox functionality
def checkbox_toggle(sender, app_data, user_data):
    dpg.configure_item("linear fput", show=app_data)

with dpg.window(label="Simulation window", pos=(450,0), width=700,height=650):
    dpg.add_checkbox(label="Show Linear", tag="linear_bool", callback=checkbox_toggle, default_value=True)
    
    dpg.add_button(label='Play', callback=start_simulation)
    dpg.add_button(label='Pause', callback=stop_simulation)
    
    with dpg.plot(label="Fixed-end String Simulation", height=250, width=600):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Node x-position", tag="x_axis")

        with dpg.plot_axis(dpg.mvYAxis, label="Node y-position", tag='y_axis'):
            dpg.add_line_series(x, q_linear, label="Linear FPUT", tag="linear fput")
            dpg.add_line_series(x, q_nonlinear, label="Non-Linear FPUT", tag="nonlinear fput")

        dpg.set_axis_limits("y_axis", -y_lim, y_lim)
        dpg.set_axis_limits("x_axis", 0,1)


    with dpg.plot(label="Energy of string over time", height=250, width=600):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="time axis", tag='time_axis')

        with dpg.plot_axis(dpg.mvYAxis, label="energy_value", tag='energy_axis'):
            dpg.add_line_series(x_time, linear_energy, label="Linear FPUT", tag="linear fput energy")
            dpg.add_line_series(x_time, nonlinear_energy, label="Non-Linear FPUT", tag="nonlinear fput energy")
            dpg.add_line_series(x_time, [initial_energy for i in range(100)], label="initial energy", tag='reference energy')
            
        dpg.set_axis_limits("energy_axis", 0, 0.5)
        dpg.set_axis_limits("time_axis", 0,100)

# Render loop
def render_loop():
    global FRAME, frame_update_rate
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        if RUNNING and (FRAME % frame_update_rate == 0): 
            update_linear_model()
            update_nonlinear_model()
            update_linear_energy()
            update_non_linear_energy()
            # dynamic_y_axis()
            dynamic_energy_axis()
            # Prevent overflow error
            FRAME = 0
        FRAME += 1

dpg.setup_dearpygui()
dpg.show_viewport()
render_loop()
dpg.destroy_context()