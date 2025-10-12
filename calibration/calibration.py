import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ximea import xiapi
import cv2
import threading
from screeninfo import get_monitors
import time

def generate_frames(N, filepath, ext = ".bmp", resolution = (800, 600), value_type = np.uint8):
    '''------------------------------------------------------------------------------------
    Generate N grayscale .bpm files
    Inputs:
       N (int): number of calibration frames to generate
       filepath (str): the filepath where the .bpm files will be saved. Default is 'calibration
       ext (str) : file extension
       resolution (tuple): the resolution of the calibration frames (width, height). Default is (800, 600)
       value_type (ndtype): numpy type for the bit depth of the calibration frames. Default is 8 bits.
    Returns:
        frames: ndarray, N x width x height array containing the grayscale values for each frame.
    To do:
    Input checks
    ------------------------------------------------------------------------------------'''
    
    frames = np.ones((N, resolution[0], resolution[1]), dtype=value_type)
    greys = np.linspace(0, np.iinfo(value_type).max, N)

    frames = frames*greys[:, None, None]

    i = 0
    for i in range(N):
        img = Image.fromarray(frames[i,:,:].astype(value_type))
        img.save(filepath+str(i)+ext)



class FullscreenDisplay:
    '''------------------------------------------------------------------------------------
    Generate N grayscale .bpm files
    
    To to:
    ? have to consult
    ------------------------------------------------------------------------------------'''
    def __init__(self, image_path, monitor_index=1):
        '''------------------------------------------------------------------------------------
        Initialisation for displaying image on a specific display with a separate thread
        Inputs:
            image_path (str): path of the 1st image to show
            monitor_index (int): index of th emonitor on which to display (0=main, default = secondary)
        Returns:
            -
        To to:
        TODO: ? have to consult
        TODO: implement input check
        ------------------------------------------------------------------------------------'''
        self.image_path = image_path
        self.monitor_index = monitor_index
        self.window_name = "Fullscreen Display"
        self._running = False
        self._thread = None
        self._lock = threading.Lock()   # Lock() is used to prevent two threads fronm accessing the same shared variable
        self._current_image = None

    def _load_image(self, path, monitor):
        '''------------------------------------------------------------------------------------
        Loads image
        Inputs:
            path (str): path of the 1st image to show
            monitor (int): index of the monitor on which the image will be to displayed
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
            # TODO: make it so that it stops the programm
        img = cv2.resize(img, (monitor.width, monitor.height))
        return img

    def _display_loop(self):
        '''------------------------------------------------------------------------------------
        Loop to display the image without interuptions
        Inputs:
            -
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        monitors = get_monitors()
        if self.monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {self.monitor_index} out of range. Found {len(monitors)} monitors.")
            # TODO: make it so that it stops the programm
        monitor = monitors[self.monitor_index]

        # initial image
        self._current_image = self._load_image(self.image_path, monitor)

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(self.window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        #Comented block would work if a frequent refresh was needed. In htis case, it is better to not have the window blink by being opened and clased all the time
        while self._running:
            with self._lock:
                img = self._current_image.copy()
            cv2.imshow(self.window_name, img)
            # keep UI responsive. Might make the screen blink
            if cv2.waitKey(1) == 33: # '"!"' manually closes the window (in case of emergency ?)
                break
                # TODO: make it so that it stops the whole programm ?
        cv2.destroyWindow(self.window_name)
        # img = self._current_image.copy()
        # cv2.imshow(self.window_name, img)

    def show(self):
        '''------------------------------------------------------------------------------------
        Function to display the 1st image
        Inputs:
            -
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._display_loop, daemon=True)
            self._thread.start()

    def update_image(self, new_image_path):
        '''------------------------------------------------------------------------------------
        Function to update the image
        Inputs:
            new_image_path (str) : path of the new image
        Returns:
            -
        To to:
        TODO: implement input checks
        ------------------------------------------------------------------------------------'''
        monitors = get_monitors()
        monitor = monitors[self.monitor_index]
        new_img = self._load_image(new_image_path, monitor)
        with self._lock: # Lock() is used to prevent two threads fronm accessing the same shared variable
            self._current_image = new_img

    def close(self):
        '''------------------------------------------------------------------------------------
        Function to close running threads and close open windows
        Inputs:
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join()
            cv2.destroyAllWindows()
            print("Fullscreen display closed.")



class Acquisition:
    '''------------------------------------------------------------------------------------
    Initialisation for acquisition on a separate thread

    To to:
    TODO: ? have to consult
    ------------------------------------------------------------------------------------'''
    def __init__(self, exposure, save_path, ext ='.bmp', ev_id=0):
        '''------------------------------------------------------------------------------------
        Initialisation for acquisition on a separate thread
        Inputs:
            exposure (int): exposure time in us
            save_path (str): save path of the acquired images
            ext (str): extenson in which to save the images
        Returns:
            -
        To to:
        TODO: ? have to consult
        TODO: implement input check
        ------------------------------------------------------------------------------------'''
        self._exposure = exposure
        self._save_path = save_path
        self._cam =  xiapi.Camera(dev_id=0)
        self._img = xiapi.Image()
        self._ext = ext
        self._running = False
        self._thread = None
        self._index = 0

        # Connecting to the camera
        print("Opening camera...")
        self._cam.open_device_by_SN('CIMAU2430046')
        print("Camera open (if it worked)")

        #setting some parameters
        sensor_bit_depth = self._cam.set_sensor_bit_depth('XI_BPP_12')
        sensor_bit_depth = self._cam.get_sensor_bit_depth()
        print('Sensor bit depth set to: ', sensor_bit_depth)

        self._cam.set_gain(0)
        print('Gain set to: ', self._cam.get_gain())

        width = self._cam.get_width()
        height = self._cam.get_height()
        print('image size: ', width,'x',height)

        self._cam.set_exposure(self._exposure) # Exposure time in us
        print('Exposure set to: ', self._cam.get_exposure(), 'us')
        self._cam.start_acquisition()

    def acquire(self, index):
        '''------------------------------------------------------------------------------------
        Initialisation for acquisition on a separate thread
        Inputs:
            index (int): index to appent onto filename
        Returns:
            -
        To to:
        TODO: implement input check
        ------------------------------------------------------------------------------------'''
        if not self._running:
            self._thread = threading.Thread(target=self._get, kwargs={'index': index}, daemon=True)
            self._thread.start()
            self._thread.join()

    def _get(self, index):
        self._cam.get_image(self._img)
        Image.fromarray(self._img.get_image_data_numpy()).save(self._save_path+str(index)+self._ext)
        self._index +=1
    def terminate(self):
        '''------------------------------------------------------------------------------------
        Initialisation for acquisition on a separate thread
        Inputs:
            -
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join()

        self._cam.stop_acquisition()
        self._cam.close_device()