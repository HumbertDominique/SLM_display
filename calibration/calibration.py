import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ximea import xiapi
import cv2
import threading
import multiprocessing
from screeninfo import get_monitors
from astropy.io import fits
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
        self._process = None
        self._manager = multiprocessing.Manager()
        self._running_flag = self._manager.Value('b', True)  # boolean flag for loop control

        self._current_image = None

    @staticmethod
    def display_loop(shared_dict, image_path, window_name, monitor_index, running_flag):
        monitors = get_monitors()
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} out of range. Found {len(monitors)} monitors.")
        monitor = monitors[monitor_index]

        # Load initial image
        img = cv2.imread(image_path)
        shared_dict['current_image'] = img

        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while running_flag.value:
            current_img = shared_dict['current_image'].copy()
            cv2.imshow(window_name, current_img)
            if cv2.waitKey(1) == 33:  # emergency close
                break
            
        cv2.destroyWindow(window_name)

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



    def show(self):
        '''------------------------------------------------------------------------------------
        Function to display the 1st image using threading
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

    def show_mp(self):
        '''------------------------------------------------------------------------------------
        Function to display the 1st image using multiprocessing
        Inputs:
            -
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        if not self._running:
            self._running = True
            self._shared_dict = self._manager.dict()
            

            self._process = multiprocessing.Process(
                target=FullscreenDisplay.display_loop,
                args=(self._shared_dict, self.image_path, self.window_name, self.monitor_index, self._running_flag),
                daemon=True
            )   # Requires the @staticmethod in order not to pass the whole self with unpickable elements if used with "self.display_loop" . "FullscreenDisplay.display_loop" could be used instead
            self._process.start()

    def update_image(self, new_image_path):
        '''------------------------------------------------------------------------------------
        Function to update the image with threading
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

    def update_image_mp(self, new_image_path):
        '''------------------------------------------------------------------------------------
        Function to update the image with multiprocessing
        Inputs:
            new_image_path (str) : path of the new image
        Returns:
            -
        To to:
        TODO: implement input checks
        ------------------------------------------------------------------------------------'''
        monitors = get_monitors()
        monitor = monitors[self.monitor_index]
        # Load new image
        new_img = self._load_image(new_image_path, monitor)
        if new_img is None:
            raise ValueError(f"Failed to load image at {new_image_path}")

    # Update the shared image in the multiprocessing loop
        self._shared_dict['current_image'] = new_img

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

            if hasattr(self, "_running_flag"):
               self._running_flag.value = False # in order fo the su-process to terminate appropriately and not have som code still running (equivalent to thread(...,..., deamon = True))

            # Wait for the process to finish
            if hasattr(self, "_process") and self._process is not None:
                self._process.join()
                self._process = None

        cv2.destroyAllWindows()
        print("Fullscreen display closed.")

    # def close_mp(self):
    #     '''------------------------------------------------------------------------------------
    #     Function to close running processes and close open windows
    #     Inputs:
    #     Returns:
    #         -
    #     To to:
    #     ------------------------------------------------------------------------------------'''
    #     if self._running:
    #         self._running = False
    #         # Stop the loop in the child process
    #         if hasattr(self, "_running_flag"):
    #            self._running_flag.value = False # in order fo the su-process to terminate appropriately and not have som code still running (equivalent to thread(...,..., deamon = True))

    #         # Wait for the process to finish
    #         if hasattr(self, "_process") and self._process is not None:
    #             self._process.join()
    #             self._process = None

    #     # Clean up windows
    #     cv2.destroyAllWindows()
    #     print("Fullscreen display closed.")


class Acquisition:
    '''------------------------------------------------------------------------------------
    Initialisation for acquisition on a separate thread

    To to:
    TODO: ? have to consult
    ------------------------------------------------------------------------------------'''
    def __init__(self, exposure, save_path, ext ='.tif', ev_id=0):
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
        self._cam.set_sensor_bit_depth('XI_BPP_12')#XI_MONO16
        self._cam.set_imgdataformat('XI_MONO16')
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


    def get(self, index):
        self._cam.get_image(self._img)
        
        cv2.imwrite(self._save_path+str(index)+self._ext,self._img.get_image_data_numpy())
    
        # img.options().save_to_16bit=True
        # img.save(self._save_path+str(index)+self._ext)
        # imageio.imwrite('result.png', im.astype(np.uint16))
        self._index +=1
        
    def terminate(self):
        '''------------------------------------------------------------------------------------
        terminates camera connection
        Inputs:
            -
        Returns:
            -
        To to:
        ------------------------------------------------------------------------------------'''
        self._cam.stop_acquisition()
        self._cam.close_device()