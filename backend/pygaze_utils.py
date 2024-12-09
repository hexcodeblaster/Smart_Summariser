import pygaze
from pygaze.libscreen import Display, Screen
from pygaze.eyetracker import EyeTracker
from pygaze import libtime

# Initialize the screen and eye tracker
disp = Display()
scr = Screen()
tracker = EyeTracker(disp)

# Calibrate the eye tracker
print("Calibrating...")
tracker.calibrate()

# Start the tracking session
print("Tracking eye positions. Press 'q' to quit.")
tracker.start_recording()

try:
    while True:
        # Get current eye position
        gaze_pos = tracker.sample()
        print(f"Current gaze position: {gaze_pos}")

        # Display a visual dot at gaze position (for debugging/feedback)
        scr.clear()
        scr.draw_fixation(fixtype='dot', pos=gaze_pos, diameter=10)
        disp.fill(scr)
        disp.show()

        # Break loop on keypress 'q'
        key, _ = libtime.expcheck()
        if key == 'q':
            break
finally:
    # Clean up
    tracker.stop_recording()
    disp.close()
    print("Eye-tracking session ended.")
