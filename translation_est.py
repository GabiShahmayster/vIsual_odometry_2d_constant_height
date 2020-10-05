import os
from typing import Optional, Tuple, List, Type
import numpy as np
from cv2 import cvtColor, COLOR_BGR2GRAY, imshow, waitKey, imread, ORB_create, ORB
from recordclass import RecordClass
from abc import ABC, abstractmethod
from enum import Enum


class ProcessedFrameData(RecordClass):
    """
    This object aggregates frame + extracted features (keypoints and descriptors) into a single data object
    """
    frame: np.ndarray
    keypoints: List[Tuple]
    descriptors: List[object]

    @classmethod
    def build(cls,
              frame: np.ndarray,
              keypoints: List[Tuple],
              descriptors: List[object]):
        return ProcessedFrameData(frame,
                                  keypoints,
                                  descriptors)


class FeaturesExtractorAbstractBase(ABC):
    """
    This is an abstract base class for a features extractor
    """

    @abstractmethod

    feature_extractor: object

    @abstractmethod
    def extract_features(self) -> ProcessedFrameData:
        """
        This method extracts features from a frame
        Returns
        -------
        """


class OrbFeaturesExtractor(FeaturesExtractorAbstractBase):
    """
    This class implements an ORB features exractor
    """
    feature_extractor: ORB

    DEFAULT_nfeatures = None
    DEFAULT_scaleFactor = None
    DEFAULT_nlevels = None
    DEFAULT_edgeThreshold = None

    def __init__(self, nfeatures: int = None,
                 scaleFactor=None,
                 nlevels: int = None,
                 edgeThreshold: float = None):
        if nfeatures is None:
            nfeatures = self.DEFAULT_nfeatures
        if scaleFactor is None:
            scaleFactor = self.DEFAULT_scaleFactor
        if nlevels is None:
            nlevels = self.DEFAULT_nlevels
        if edgeThreshold is None:
            edgeThreshold = self.DEFAULT_edgeThreshold

        self.feature_extractor = ORB_create(nfeatures=nfeatures,
                                            scaleFactor=scaleFactor,
                                            nlevels=nlevels,
                                            edgeThreshold=edgeThreshold)

    def extract_features(self) -> ProcessedFrameData:
        keypoints, descriptors = self.feature_extractor.detectAndCompute()
        return ProcessedFrameData.build(frame=,
                                        keypoints=keypoints,
                                        descriptors=descriptors)


class MotionEstimationStatus(Enum):
    """
    This enum indicates the motion estimation algorithm status
    """
    WAITING_FOR_FIRST_FRAME = 0
    UPDATED_MOTION_UNAVAILABLE = 1


class RobotHorizontalMotionEstimator:
    """
    This class tracks the horizontal motion of a robot, navigating above a ground-plane,
    using an incoming video stream from a down-facing camera
    """
    features_extractor: Type[FeaturesExtractorAbstractBase]
    previous_frame_data: Optional[ProcessedFrameData]
    robot_height_in_meter: float
    current_robot_position: Optional[Tuple]
    motion_estimation_status: MotionEstimationStatus

    def __init__(self, robot_height_in_meter: float):
        self.features_extractor = OrbFeaturesExtractor()
        self.robot_height_in_meter = robot_height_in_meter
        self.init_motion_estimation()

    def init_motion_estimation(self):
        """
        This method initializes the motion estimation block
        """
        self.previous_frame_data = None
        self.motion_estimation_status = MotionEstimationStatus.WAITING_FOR_FIRST_FRAME

    def set_new_frame(self, raw_frame: np.ndarray):
        """
        This method passes a new frame to the motion estimation class
        Parameters
        ----------
        raw_frame

        Returns
        -------

        """
        frame: np.ndarray = self.prepare_frame(frame=raw_frame)
        self.update_motion_using_new_frame(frame=frame)

    def prepare_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        This method prepares the received frame for subsequent processing
        Returns
        -------

        """
        return cvtColor(frame, COLOR_BGR2GRAY)

    def process_frame(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts features from a frame and returns an aggregated ProcessedFrameData object
        """
        keypoints, descriptors = self.features_extractor.detectAndCompute(frame, None)
        return ProcessedFrameData.build(frame=frame,
                                        keypoints=keypoints,
                                        descriptors=descriptors)

    def update_motion_using_new_frame(self, frame: np.ndarray) -> Optional[Tuple]:
        """
        This method
        Parameters
        ----------
        frame

        Returns
        -------

        """
        processed_frame: ProcessedFrameData = self.process_frame(frame=frame)
        if self.previous_frame_data is None:
            # first frame, cannot estimate motion
            self.previous_frame_data = processed_frame
            self.motion_estimation_status = MotionEstimationStatus.UPDATED_MOTION_UNAVAILABLE
            return None

        self.current_robot_position =

    def update_motion(self):
        """
        This method estimates the horizontal translation of a robot, navigating above a ground-plance,
        given two down-facing frames captured by the robot's camera.
        The robot's altitude above the ground-plane is assumed to be known and constant
        """


        waitKey(0)

    def print_current_position(self):
        """
        This method prints the current position of the robot, if available
        """
        if self.current_robot_position is None:
            print("")


if __name__ == "__main__":
    current_dir: str = os.getcwd()
    list_of_frame_paths: List[str] = list()
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_1.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_2.png'))

    robot_height_in_meter = 2.5
    robot_motion_estimator: RobotHorizontalMotionEstimator = RobotHorizontalMotionEstimator(
        robot_height_in_meter=robot_height_in_meter)

    for frame_path in list_of_frame_paths:
        frame: Optional[np.ndarray] = imread(frame_path)
        if frame is not None:
            robot_motion_estimator.set_new_frame()
            robot_motion_estimator.print_current_position()
        else:
            print("cannot read frame")
            exit(1)

    translation: Optional[Tuple] = estimate_horizontal_translation(frame_1_path=frame_1_path,
                                                                   frame_2_path=frame_2_path,
                                                                   robot_height_in_meter=robot_height_in_meter)
    if translation is not None:
        print("X translation = {0:f} Y translation = {1:f}".format(translation[0], translation[1]))
    else:
        print("Could not estimated robot translation")
    exit(0)
