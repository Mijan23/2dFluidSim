import pygame
import numpy as np

# Simulation Parameters
N = 50  # Increased Grid size for more depth
dt = 0.1  # Time step
diff = 0.000000001  # Diffusion rate
visc = 0.000000001  # Viscosity
force = 500.0  # Increased strength of user input
source = 50000.0  # Further increased maximum density added per frame

# Initialize fields
density = np.zeros((N, N))
velocity_x = np.zeros((N, N))
velocity_y = np.zeros((N, N))

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((512, 512))
clock = pygame.time.Clock()
running = True

# Index helper function
def IX(x, y):
    return x, y

# Add source to density and velocity
def add_source(x, y, amount):
    density[x, y] += amount
    velocity_x[x, y] += np.random.uniform(-force, force)
    velocity_y[x, y] += np.random.uniform(-force, force)

# Diffusion step
def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N-2) * (N-2)
    for _ in range(20):  # Gauss-Seidel relaxation
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / (1 + 4 * a)

# Advection step
def advect(b, d, d0, vx, vy, dt):
    for i in range(1, N-1):
        for j in range(1, N-1):
            x, y = i - dt * vx[i, j], j - dt * vy[i, j]
            x, y = max(0.5, min(N-1.5, x)), max(0.5, min(N-1.5, y))
            i0, j0 = int(x), int(y)
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

# Projection step
def project(vx, vy, p, div):
    div[1:-1, 1:-1] = -0.5 * (vx[2:, 1:-1] - vx[:-2, 1:-1] + vy[1:-1, 2:] - vy[1:-1, :-2]) / N
    p.fill(0)
    for _ in range(20):
        p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
    vx[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * N
    vy[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * N

# Main Loop
while running:
    screen.fill((0, 0, 0))
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Mouse interaction
    mouse_pressed = pygame.mouse.get_pressed()
    if mouse_pressed[0]:
        mx, my = pygame.mouse.get_pos()
        i, j = int(mx / 512 * N), int(my / 512 * N)
        add_source(i, j, source)  # Further increased maximum intensity on press
    
    # Fluid simulation steps
    diffuse(0, velocity_x, velocity_x, visc, dt)
    diffuse(0, velocity_y, velocity_y, visc, dt)
    project(velocity_x, velocity_y, np.zeros((N, N)), np.zeros((N, N)))
    advect(0, velocity_x, velocity_x, velocity_x, velocity_y, dt)
    advect(0, velocity_y, velocity_y, velocity_x, velocity_y, dt)
    project(velocity_x, velocity_y, np.zeros((N, N)), np.zeros((N, N)))
    diffuse(0, density, density, diff, dt)
    advect(0, density, density, velocity_x, velocity_y, dt)
    
    # Render fluid density
    for i in range(N):
        for j in range(N):
            d = min(255, int(density[i, j] * 255 / (source / 10)))
            color = (255, d, d)  # Brighter red fluid emission
            pygame.draw.rect(screen, color, (i * 512 / N, j * 512 / N, 512 / N, 512 / N))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
