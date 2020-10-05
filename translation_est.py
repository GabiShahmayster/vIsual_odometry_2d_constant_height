import os
from typing import Optional, Tuple, List, Type
import numpy as np
from cv2 import cvtColor, COLOR_BGR2GRAY, imshow, waitKey, imread, ORB_create, ORB
from recordclass import RecordClass
from abc import ABC, abstractmethod


class FrameFeatures(RecordClass):
    """
    This object aggregates frame features data (keypoints and descriptors) into a single data object
    """
    keypoints: List[Tuple]
    descriptors: List[object]

    @classmethod
    def build(cls, keypoints: List[Tuple],
              descriptors: List[object]):
        return FrameFeatures(keypoints,
                             descriptors)


class FeaturesExtractorAbstractBase(ABC):
    """
    This is an abstract base class for a features extractor
    """

    @abstractmethod

    feature_extractor: object

    @abstractmethod
    def extract_features(self) -> FrameFeatures:
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

    def extract_features(self) -> FrameFeatures:
        keypoints, descriptors = self.feature_extractor.detectAndCompute()
        return FrameFeatures.build(keypoints=keypoints,
                                   descriptors=descriptors)


class RobotHorizontalMotionEstimator:
    """
    This class tracks the horizontal motion of a robot, navigating above a ground-plane,
    using an incoming video stream from a down-facing camera
    """
    features_extractor: Type[FeaturesExtractorAbstractBase]
    previous_frame: Optional[np.ndarray]
    robot_height_in_meter: float

    def __init__(self, robot_height_in_meter: float):
        self.features_extractor = OrbFeaturesExtractor()
        self.previous_frame = None
        self.robot_height_in_meter = robot_height_in_meter

    def set_new_frame(self, new_frame: np.ndarray):
        """
        This method passes a new frame to the motion estimation class
        Parameters
        ----------
        new_frame

        Returns
        -------

        """
        processed_frame: np.ndarray = self.prepare_frame(frame=new_frame)
        self.update_motion_using_new_frame(new_frame=processed_frame)

    def prepare_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        This method prepares the received frame for subsequent processing
        Returns
        -------

        """
        return cvtColor(frame, COLOR_BGR2GRAY)

    def update_motion_using_new_frame(self, new_frame: np.ndarray) -> Optional[Tuple]:
        """
        This method
        Parameters
        ----------
        new_frame

        Returns
        -------

        """
        if self.previous_frame is None:
            # first frame, cannot estimate motion
            self.previous_frame = new_frame
            return None

    def estimate_horizontal_translation(self,
                                        frame_1_path: str,
                                        frame_2_path: str,
                                        robot_height_in_meter: float) -> Optional[Tuple]:
        """
        This method estimates the horizontal translation of a robot, navigating above a ground-plance,
        given two down-facing frames captured by the robot's camera.
        The robot's altitude above the ground-plane is assumed to be known and constant
        Parameters
        ----------
        frame_1_path - absolute path to 1-st frame
        frame_2_path - absolute path to 2-nd frame
        robot_height_in_meter - robot height [m]
        Returns
        -------
        """

        # TODO assert or convert to BGR color-space

        imshow('first frame', frame_1_gray)
        imshow('second frame', frame_2_gray)

        # construct ORB features detector
        orb_detector = ORB_create()

        keypoints, descriptors = orb_detector.detectAndCompute(frame_1_gray)

        waitKey(0)


if __name__ == "__main__":
    current_dir: str = os.getcwd()
    list_of_frame_paths: List[str] = list()
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_1.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_2.png'))

    robot_height_in_meter = 2.5
    robot_motion_estimator: RobotHorizontalMotionEstimator = RobotHorizontalMotionEstimator(robot_height_in_meter=robot_height_in_meter)

    for frame_path in list_of_frame_paths:

        robot_motion_estimator.set_new_frame()

    translation: Optional[Tuple] = estimate_horizontal_translation(frame_1_path=frame_1_path,
                                                                   frame_2_path=frame_2_path,
                                                                   robot_height_in_meter=robot_height_in_meter)
    if translation is not None:
        print("X translation = {0:f} Y translation = {1:f}".format(translation[0], translation[1]))
    else:
        print("Could not estimated robot translation")
    exit(0)
