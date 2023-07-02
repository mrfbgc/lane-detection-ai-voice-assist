__author__ = "Erdi Örün, Mehmet Arif Bağcı, İlyas Çavdır"
__copyright__ = "Copyright 2023, Trakya University"
__version__ = "0.9.0"
__status__ = "Development"

import time
import numpy as np
import cv2
import random
import pygame
import pyfiglet
import glob
import os
import sys
import lane
try:
    sys.path.append(glob.glob('./dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


ascii_banner = pyfiglet.figlet_format("Lane Detection")
print(ascii_banner)
print("version: " + __version__ + " \nstatus: " + __status__)


IM_WIDTH = 1280
IM_HEIGHT = 720
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def process_image(image):

    frame = np.array(image.raw_data)

    frame = frame.reshape((image.height, image.width, 4))

    frame = frame[:, :, :3]

    processed_frame = process_frame_with_lanes(frame)

    display_frame(processed_frame)


def process_frame_with_lanes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    height, width = frame.shape[:2]
    roi_vertices = np.array([
        (120, height),
        (width * 0.48, height * 0.52),
        (width * 0.53, height * 0.52),
        (width-120, height)
    ], np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    frame_roi_visualized = cv2.polylines(
        frame.copy(), [roi_vertices], True, (0, 255, 0), thickness=2)

    lines = cv2.HoughLinesP(
        masked_edges, rho=2, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    lane_image = np.zeros_like(frame)
    road_image = np.zeros_like(frame)

    lane_color = (0, 0, 255)
    road_color = (0, 255, 0)

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < -0.3:
                    if length > 50:
                        left_lines.append(line)
                elif slope > 0.3:
                    if length > 50:
                        right_lines.append(line)

    left_x1 = left_x2 = int(width / 2)
    left_y2 = int(height * 0.6)

    if left_lines:
        left_points = np.array(
            [[x1, y1] for line in left_lines for x1, y1, x2, y2 in line])
        left_vx, left_vy, left_x0, left_y0 = cv2.fitLine(
            left_points, cv2.DIST_L2, 0, 0.01, 0.01)
        left_slope = left_vy / left_vx if left_vx != 0 else np.inf
        left_intercept = left_y0 - left_slope * left_x0
        left_y1 = height
        if left_slope != 0:
            left_x1 = int((left_y1 - left_intercept) / left_slope)
            left_y2 = int(height * 0.6)
            left_x2 = int((left_y2 - left_intercept) / left_slope)

        cv2.line(lane_image, (left_x1, left_y1),
                 (left_x2, left_y2), lane_color, 10)

    right_x1 = int(width * 0.8)
    right_x2 = int(width * 0.8)
    right_y2 = int(height * 0.6)

    if right_lines:
        right_points = np.array(
            [[x1, y1] for line in right_lines for x1, y1, x2, y2 in line])
        right_vx, right_vy, right_x0, right_y0 = cv2.fitLine(
            right_points, cv2.DIST_L2, 0, 0.01, 0.01)
        right_slope = right_vy / right_vx if right_vx != 0 else np.inf
        right_intercept = right_y0 - right_slope * right_x0
        right_y1 = height

        if right_slope != 0:
            right_x1 = int((right_y1 - right_intercept) / right_slope)
            right_y2 = int(height * 0.6)
            right_x2 = int((right_y2 - right_intercept) / right_slope)

        cv2.line(lane_image, (right_x1, right_y1),
                 (right_x2, right_y2), lane_color, 10)

    if left_lines and right_lines:
        road_polygon = np.array([
            (left_x1, left_y1),
            (right_x1, right_y1),
            (right_x2, right_y2),
            (left_x2, left_y2)
        ], np.int32)

        cv2.fillPoly(road_image, [road_polygon], road_color)

    left_point = np.array([left_x2, left_y2])
    right_point = np.array([right_x2, right_y2])
    bottom_center = np.array([width // 2, height])
    curve = np.cross(left_point - bottom_center, right_point - bottom_center)
    curve = curve / (np.linalg.norm(left_point - bottom_center)
                     * np.linalg.norm(right_point - bottom_center))

    curve_text = f"Curve: {curve:.2f}"
    frame = np.array(frame)
    frame = cv2.putText(frame, curve_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    result = cv2.addWeighted(frame, 1, lane_image, 0.5, 0)
    result = cv2.addWeighted(result, 1, road_image, 0.5, 0)

    result_with_roi_visualized = cv2.addWeighted(
        result, 1, frame_roi_visualized, 0.5, 0)

    return result_with_roi_visualized


def display_frame(frame):
    frame_flipped = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)

    image_surface = pygame.surfarray.make_surface(frame_rgb)
    screen.blit(image_surface, (0, 0))
    pygame.display.update()


def process_image2(image):

    lane_obj = lane.Lane(orig_frame=image)

    lane_line_markings = lane_obj.get_line_markings()

    lane_obj.plot_roi(plot=False)

    warped_frame = lane_obj.perspective_transform(plot=False)

    histogram = lane_obj.calculate_histogram(plot=False)

    # Find lane line pixels using the sliding window method
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
        plot=False)

    # Fill in the lane line
    lane_obj.get_lane_line_previous_window(
        left_fit, right_fit, plot=False)

    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

    # Calculate lane line curvature (left and right lane lines)
    lane_obj.calculate_curvature(print_to_terminal=False)

    # Calculate center offset
    lane_obj.calculate_car_position(print_to_terminal=False)

    # Display curvature and center offset on image
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=False)

    return frame_with_lane_lines2


actor_list = []

client = carla.Client("localhost", 2000)
client.set_timeout(200.0)

# world = client.load_world("Town04OPT")
world = client.get_world()
weather = carla.WeatherParameters(
    cloudiness=100.0,
    precipitation=0.1,
    sun_altitude_angle=10.0)

world.set_weather(weather)

blueprint_library = world.get_blueprint_library()

# araba secimi
vehicle_bp = blueprint_library.filter("model3")[0]

# spawn
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Set up the vehicle's initial state
vehicle_control = carla.VehicleControl()

spawn_point = carla.Transform(carla.Location(x=1.9, z=1.5))
# spawn_point = carla.Transform(carla.Location(x=2.5, z=1.1))

cam_bp = blueprint_library.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
cam_bp.set_attribute("fov", "100")

pygame.init()
clock = pygame.time.Clock()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

camera_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)

actor_list.append(camera_sensor)

camera_sensor.listen(process_image)


def game_loop():
    while True:
        # Start the Pygame event loop
        vehicle.apply_control(vehicle_control)

        # Handle keyboard events
        handle_events(clock.get_time())

        # Tick the simulation and advance the Pygame clock
        world.tick()
        clock.tick_busy_loop(60)


def handle_events(milliseconds):
    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 +
                   velocity.z**2)**0.5  # convert to km/h
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                vehicle_control.throttle = 0.7
            elif event.key == pygame.K_DOWN:
                vehicle_control.brake = 1.0
            elif event.key == pygame.K_LEFT:
                vehicle_control.steer = -0.2
            elif event.key == pygame.K_RIGHT:
                vehicle_control.steer = 0.2
            elif event.key == pygame.K_q:
                vehicle_control.gear = -1

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                vehicle_control.throttle = 0.0
            elif event.key == pygame.K_DOWN:
                vehicle_control.brake = 0.0
            elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                vehicle_control.steer = 0.0


game_loop()
