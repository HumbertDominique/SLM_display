from calibration import generate_frames, FullscreenDisplay, Acquisition
import time
import keyboard

SN = 'CIMAU2430046'
path1 = "data/calibration_"
path2 = "data/measurements/mes_"
ext = ".bmp"
monitor_index = 2 # 0:main, 1:secondary, 3:tertiary, ...

exposure = 20000
exposure_s = exposure*1e-6
Stabilisation_time = 5e-2
N = 50
HYSTERESIS = True
hysteresis_index_offset = 1000

index = 0

generate_frames(N, path1)

RUNNING = True

print(path1+str(index)+".bmp")
display = FullscreenDisplay(path1+str(index)+".bmp", monitor_index)
# display.show()    # For use with threading
display.show_mp()
time.sleep(Stabilisation_time)

print("Starting acquisition")
acquisition = Acquisition(exposure, path2)
print("Acquisition ready")
tic = time.time()
tac = tic
print("Press \'e\' for emergency exit")
while RUNNING:
    if tac-tic >= Stabilisation_time: 
        acquisition.get(index)  # makes the programm wait on it
        if index == N - 1:
            RUNNING = False
        else :
            # display.update_image(path1+str(index)+ext)
            display.update_image_mp(path1+str(index)+ext)
            tic = tac
            index += 1
            
    tac = time.time()
    if keyboard.is_pressed('e'):
        print('Process abrubptly terminated. Restart kernel.')
        RUNNING = False
        HYSTERESIS = False
    time.sleep(0.001)
    
tic = time.time()
tac = tic
while HYSTERESIS:
    if tac-tic >= Stabilisation_time:
        acquisition.get(index+hysteresis_index_offset)# makes the programm wait on it
        if index == 0:
            HYSTERESIS = False
        else :
            display.update_image_mp(path1+str(index)+ext)
            tic = tac
            index -= 1
    

    tac = time.time()
    if keyboard.is_pressed('e'):
        print('Process abrubptly terminated. Restart kernel.')
        RUNNING = False
    time.sleep(0.001)

print("Terminating processes")
acquisition.terminate()
display.close()
