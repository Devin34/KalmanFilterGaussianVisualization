import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


#GOAL: Find "true" robot "distance".

#Mu (μ) is the mean
#Sigma (σ²) is the variance kind of like the error (ex: increasing variance in sensor means the measurement is less accurate or trustworthy)

# x-axis
x = np.linspace(-10, 10, 1000)

def gaussian(x, mu, sigma):
    #Function for Gaussian distribution
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2) 

# Initial Values
mu1, sigma1 = 0, 1       # prior predicted dist. 
mu2, sigma2 = 2, 1.5     # dist. based on most recent measurement

# Figure
fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(left=0.15, bottom=0.35)

# Initial Gaussians
y1 = gaussian(x, mu1, sigma1)
y2 = gaussian(x, mu2, sigma2)

# Kalman Update
def kalman_update(mu_p, sigma_p, mu_m, sigma_m):

    #Calculate Kalman Gain (E_est/E_est + E_mea)
    k = sigma_p**2 / (sigma_p**2 + sigma_m**2)

    #Calculate new dist. estimate
    mu_post = mu_p + k * (mu_m - mu_p)

    #Calculate new dist. error (ideally should be smaller than prior and measurement sigmas since we have a better guess)
    sigma_post = np.sqrt((1 - k) * sigma_p**2)
    return mu_post, sigma_post

#Resultant Kalman Gaussian
mu3, sigma3 = kalman_update(mu1, sigma1, mu2, sigma2)
y3 = gaussian(x, mu3, sigma3)

# Lines
line1, = ax.plot(x, y1, label="Prior")
line2, = ax.plot(x, y2, label="Measurement")
line3, = ax.plot(x, y3, linewidth=2.5, label="Kalman")



ax.set_xlim(-10, 10)
ax.set_ylim(0, 0.6)
ax.set_xlabel("x")
ax.set_ylabel("Probability Density")
ax.set_title("Kalman Filter Example")
ax.legend()
ax.grid(True)

# Sliders
ax_mu1 = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_sigma1 = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_mu2 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_sigma2 = plt.axes([0.2, 0.10, 0.65, 0.03])

slider_mu1 = Slider(ax_mu1, r"$\mu_{prior}$", -5, 5, valinit=mu1)
slider_sigma1 = Slider(ax_sigma1, r"$\sigma_{prior}$", 0.1, 5, valinit=sigma1)

slider_mu2 = Slider(ax_mu2, r"$\mu_{meas}$", -5, 5, valinit=mu2)
slider_sigma2 = Slider(ax_sigma2, r"$\sigma_{meas}$", 0.1, 5, valinit=sigma2)

# Update function
def update(val):
    mu_p = slider_mu1.val
    sigma_p = slider_sigma1.val
    mu_m = slider_mu2.val
    sigma_m = slider_sigma2.val

    #Update Gaussians
    y1 = gaussian(x, mu_p, sigma_p)
    y2 = gaussian(x, mu_m, sigma_m)

    mu_post, sigma_post = kalman_update(mu_p, sigma_p, mu_m, sigma_m)
    y3 = gaussian(x, mu_post, sigma_post)

    # update lines
    line1.set_ydata(y1)
    line2.set_ydata(y2)
    line3.set_ydata(y3)

    fig.canvas.draw_idle()

#Update values when sliders change
slider_mu1.on_changed(update)
slider_sigma1.on_changed(update)
slider_mu2.on_changed(update)
slider_sigma2.on_changed(update)

plt.show()