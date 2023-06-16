import numpy as np
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2hsv
from skimage.morphology import disk
from skimage.filters.rank import gradient, median
from skimage.color import rgb2gray
from mpl_toolkits.mplot3d import Axes3D
import simpy


image_rgb = io.imread("./picture/benchmark_3.jpg")[:,:,:3] #import the your picture
plt.figure()
plt.imshow(image_rgb)
plt.show()

image_gray = rgb2gray(image_rgb)

image_median = median(image_gray, disk(3))
median_grad = gradient(image_median, disk(1))
median_grad = median_grad / np.max(median_grad)

image_hsv = rgb2hsv(image_rgb)
image_sat = image_hsv[:, :, 1]
sat_med = median(image_sat, disk(3))
sat_med_grad = gradient(sat_med, disk(1))
sat_med_grad = sat_med_grad / np.max(sat_med_grad)

universe = np.linspace(0, 1, 10)

Gray = ctrl.Antecedent(universe, 'Gray')
Saturation = ctrl.Antecedent(universe, 'Saturation')

Sketch = ctrl.Consequent(universe, 'Sketch')

names = ['low', 'medium', 'high']
Gray.automf(names=names)
Saturation.automf(names=names)
Sketch.automf(names=names)

rule1 = ctrl.Rule(antecedent=Gray['low'] & Saturation['low'], consequent=Sketch['low'])
rule2 = ctrl.Rule(antecedent=Gray['low'] & Saturation['medium'], consequent=Sketch['low'])
rule3 = ctrl.Rule(antecedent=Gray['low'] & Saturation['high'], consequent=Sketch['medium'])
rule4 = ctrl.Rule(antecedent=Gray['medium'] & Saturation['low'], consequent=Sketch['low'])
rule5 = ctrl.Rule(antecedent=Gray['medium'] & Saturation['medium'], consequent=Sketch['medium'])
rule6 = ctrl.Rule(antecedent=Gray['medium'] & Saturation['high'], consequent=Sketch['high'])
rule7 = ctrl.Rule(antecedent=Gray['high'] & Saturation['low'], consequent=Sketch['medium'])
rule8 = ctrl.Rule(antecedent=Gray['high'] & Saturation['medium'], consequent=Sketch['high'])
rule9 = ctrl.Rule(antecedent=Gray['high'] & Saturation['high'], consequent=Sketch['high'])

system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
sim = ctrl.ControlSystemSimulation(system)

upsampled = np.linspace(0, 1, 21)
x, y = np.meshgrid(upsampled, upsampled)
z = np.zeros_like(x)
for i in range(21):
    for j in range(21):
        sim.input['Gray'] = x[i, j]
        sim.input['Saturation'] = y[i, j]
        sim.compute()
        z[i, j] = sim.output['Sketch']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
ax.view_init(30, 200)

sim.input['Gray'] = x[i, j]
sim.input['Saturation'] = y[i, j]
sim.compute()
z[i, j] = sim.output['Sketch']

output_image = np.zeros(shape=np.shape(image_gray))
for i in range(np.shape(image_gray)[0]):
    for j in range(np.shape(image_gray)[1]):
        sim.input['Gray'] = median_grad[i, j]
        sim.input['Saturation'] = sat_med_grad[i, j]
        sim.compute()
        output_image[i, j] = sim.output['Sketch']

plt.figure()
plt.imshow(output_image < 0.25, cmap="gray", interpolation="nearest")
plt.show()

