import os
from typing import Optional, Tuple, List, Type
import numpy as np
from cv2 import cvtColor, COLOR_BGR2GRAY, imshow, waitKey, imread, ORB_create, ORB, KeyPoint, drawMarker, MARKER_CROSS, \
    COLOR_GRAY2BGR, BFMatcher, NORM_HAMMING, DMatch, drawMatches, findHomography, RANSAC, decomposeHomographyMat, \
    Rodrigues, resize, INTER_AREA, findEssentialMat, decomposeEssentialMat, solvePnPRansac, projectPoints
from recordclass import RecordClass
from abc import ABC, abstractmethod
from enum import Enum

radiansToDegrees: float = 1 / 180.0 * np.pi


class ProcessedFrameData(RecordClass):
    """
    This object aggregates frame + extracted features (keypoints and descriptors) into a single data object
    """
    frame: np.ndarray
    list_of_keypoints: List[KeyPoint]
    descriptors: np.ndarray

    @classmethod
    def build(cls,
              frame: np.ndarray,
              list_of_keypoints: List[KeyPoint],
              descriptors: np.ndarray):
        return ProcessedFrameData(frame,
                                  list_of_keypoints,
                                  descriptors)


class RobotPose(RecordClass):
    """
    This objects contains the robot pose (horizontal translation + yaw angle)
    """
    position_x_meter: float
    position_y_meter: float
    yaw_angle_deg: float

    @classmethod
    def build(cls,
              position_x_meter: float,
              position_y_meter: float,
              yaw_angle_deg: float):
        return RobotPose(position_x_meter,
                         position_y_meter,
                         yaw_angle_deg)


class FeaturesHandlerAbstractBase(ABC):
    """
    This is an abstract base class for a features extractor
    """

    features_extractor: object
    features_matcher: object

    @abstractmethod
    def extract_features(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts features from a frame
        Returns
        -------
        """
        pass

    @abstractmethod
    def match_features(self, frame_1: ProcessedFrameData, frame_2: ProcessedFrameData):
        """
        This method matches features between frames
        """


class OrbFeaturesHandler(FeaturesHandlerAbstractBase):
    """
    This class implements an ORB features exractor
    """
    features_extractor: ORB
    features_matcher: BFMatcher

    DEFAULT_DISTANCE_THRESHOLD_FOR_SUCCESSFULL_FEATURE_MATCH: int = 10

    DEFAULT_nfeatures = 500

    # DEFAULT_scaleFactor = None
    # DEFAULT_nlevels = None
    # DEFAULT_edgeThreshold = None

    # def __init__(self, nfeatures: int = None,
    #              scaleFactor=None,
    #              nlevels: int = None,
    #              edgeThreshold: float = None):
    def __init__(self):
        # if nfeatures is None:
        #     nfeatures = self.DEFAULT_nfeatures
        # if scaleFactor is None:
        #     scaleFactor = self.DEFAULT_scaleFactor
        # if nlevels is None:
        #     nlevels = self.DEFAULT_nlevels
        # if edgeThreshold is None:
        #     edgeThreshold = self.DEFAULT_edgeThreshold

        self.features_extractor = ORB_create(nfeatures=self.DEFAULT_nfeatures)

        # ORB uses binary descriptors -> use hamming norm (xor between descriptors)
        self.features_matcher = BFMatcher(NORM_HAMMING, crossCheck=True)

    def extract_features(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts ORB features
        """
        list_of_keypoints, descriptors = self.features_extractor.detectAndCompute(image=frame,
                                                                                  mask=None)
        return ProcessedFrameData.build(frame=frame,
                                        list_of_keypoints=list_of_keypoints,
                                        descriptors=descriptors)

    def match_features(self, frame_1: ProcessedFrameData, frame_2: ProcessedFrameData):
        """
        This method matches ORB features
        based on https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        """
        list_of_matches: List[DMatch] = self.features_matcher.match(queryDescriptors=frame_1.descriptors,
                                                                    trainDescriptors=frame_2.descriptors)
        return sorted(list_of_matches, key=lambda x: x.distance)

        # # Sort them in the order of their distance.
        # return


class MotionEstimationStatus(Enum):
    """
    This enum indicates the motion estimation algorithm status
    """
    MOTION_WAITING_FOR_FIRST_FRAME = 0
    MOTION_NOT_UPDATED = 1
    MOTION_OK = 2


class RobotHorizontalMotionEstimator:
    """
    This class tracks the horizontal motion of a robot, navigating above a ground-plane,
    using an incoming video stream from a down-facing camera
    """
    robot_height_in_meter: float
    camera_intrinsic: np.ndarray

    features_handler: Type[FeaturesHandlerAbstractBase]
    previous_frame_data: Optional[ProcessedFrameData]
    current_frame_number: int

    current_robot_pose: Optional[RobotPose]
    motion_estimation_status: MotionEstimationStatus

    debug_flag: bool
    MARKER_TYPE_FOR_DEBUG: int = MARKER_CROSS
    MARKER_SIZE_FOR_DEBUG: int = 5

    RESIZE_RATIO_FOR_DEBUG: int = 2

    DEFAULT_NISTER_PROB: float = .999
    DEFAULT_NISTER_THRESHOLD: float = 1.0

    def __init__(self, robot_height_in_meter: float,
                 camera_intrinsic: np.ndarray,
                 debug_flag: bool = False):
        self.features_handler = OrbFeaturesHandler()
        self.robot_height_in_meter = robot_height_in_meter
        self.camera_intrinsic = camera_intrinsic
        self.current_frame_number = 0
        self.init_motion_estimation()
        self.debug_flag = debug_flag

    def init_motion_estimation(self):
        """
        This method initializes the motion estimation block
        """
        self.previous_frame_data = None
        self.motion_estimation_status = MotionEstimationStatus.MOTION_WAITING_FOR_FIRST_FRAME
        self.current_robot_pose = None

    def set_new_frame(self, raw_frame: np.ndarray):
        """
        This method passes a new frame to the motion estimation class
        Parameters
        ----------
        raw_frame

        Returns
        -------

        """
        self.update_motion_using_new_frame(frame=self.prepare_frame(frame=raw_frame))
        self.current_frame_number += 1

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
        return self.features_handler.extract_features(frame=frame)

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
        if self.debug_flag:
            # draw features keypoints
            temp_frame: np.ndarray = cvtColor(processed_frame.frame, COLOR_GRAY2BGR)
            for keypoint in processed_frame.list_of_keypoints:
                drawMarker(img=temp_frame,
                           position=(int(keypoint.pt[0]), int(keypoint.pt[1])),
                           color=(0, 255, 0),
                           markerSize=self.MARKER_SIZE_FOR_DEBUG,
                           markerType=self.MARKER_TYPE_FOR_DEBUG)
            imshow(winname="frame #{0:d} with detected features".format(self.current_frame_number),
                   mat=self.resize_image(temp_frame))

        if self.previous_frame_data is None:
            # first frame, cannot estimate motion
            print("Robot motion estimation initialized")
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED
        else:
            self.update_motion(new_processed_frame_data=processed_frame)
        self.previous_frame_data = processed_frame

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        This method resizes an image, for visualization
        """
        scale_percent = int(100 / self.RESIZE_RATIO_FOR_DEBUG)  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        return resize(image, dim, interpolation=INTER_AREA)

    def update_motion(self, new_processed_frame_data: ProcessedFrameData):
        """
        This method estimates the horizontal translation of a robot, navigating above a ground-plance,
        given two down-facing frames captured by the robot's camera.
        The robot's altitude above the ground-plane is assumed to be known and constant
        """
        # match features
        list_of_matches: List[DMatch] = self.features_handler.match_features(frame_1=self.previous_frame_data,
                                                                             frame_2=new_processed_frame_data)
        list_of_matches = list_of_matches[:20]
        if self.debug_flag:
            imshow("feature matches", mat=self.resize_image(drawMatches(self.previous_frame_data.frame,
                                                                        self.previous_frame_data.list_of_keypoints,
                                                                        new_processed_frame_data.frame,
                                                                        new_processed_frame_data.list_of_keypoints,
                                                                        list_of_matches,
                                                                        None,
                                                                        flags=2)))

        # find homography
        # src_pts = np.float32([self.previous_frame_data.list_of_keypoints[match.queryIdx].pt for match in list_of_matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([new_processed_frame_data.list_of_keypoints[match.trainIdx].pt for match in list_of_matches]).reshape(-1, 1, 2)

        # self.get_camera_motion_using_homography(src_pts, dst_pts)
        # self.get_camera_motion_using_epipolar_constraint(src_pts, dst_pts)
        self.get_camera_motion_using_PnP(list_of_matches,
                                         new_processed_frame_data=new_processed_frame_data)
        """
        3. find homography
        4. decompose homography
        5. choose rotation & translation
        6. use known camera height to scale translation
        """

    def print_current_position(self):
        """
        This method prints the current position of the robot, if available
        """
        if self.motion_estimation_status == MotionEstimationStatus.MOTION_OK:
            print("frame #{0:d} | Robot motion: X = {1:f} [m] Y = {2:f} [m] | rotation: heading = {3:f} [deg]".format(
                self.current_frame_number,
                self.current_robot_pose.position_x_meter,
                self.current_robot_pose.position_y_meter, self.current_robot_pose.yaw_angle_deg))
        elif self.motion_estimation_status == MotionEstimationStatus.MOTION_WAITING_FOR_FIRST_FRAME:
            print("frame #{0:d} | Robot motion unavailable - waiting for 1-st frame".format(self.current_frame_number))
        elif self.motion_estimation_status == MotionEstimationStatus.MOTION_NOT_UPDATED:
            print("frame #{0:d} | Robot motion could not be updated".format(self.current_frame_number))

    def get_camera_motion_using_homography(self, src_pts, dst_pts):
        """
        https://stackoverflow.com/questions/35942095/opencv-strange-rotation-and-translation-matrices-from-decomposehomographymat
        """

        M, mask = findHomography(src_pts, dst_pts, RANSAC, 5.0)
        num_res, Rs, ts, n = decomposeHomographyMat(H=M, K=self.camera_intrinsic.astype(np.float32))

        print("-------------------------------------------\n")
        print("Estimated decomposition:\n\n")
        for i, Rt in enumerate(zip(Rs, ts)):
            R, t = Rt
            print("option " + str(i + 1))
            print("rvec = ")
            rvec, _ = Rodrigues(R)
            print(rvec * 180 / np.pi)
            print("t = ")
            print(t)

    def get_camera_motion_using_epipolar_constraint(self, src_pts, dst_pts):
        E, _ = findEssentialMat(points1=src_pts,
                                points2=dst_pts,
                                cameraMatrix=self.camera_intrinsic,
                                method=RANSAC,
                                prob=self.DEFAULT_NISTER_PROB,
                                threshold=self.DEFAULT_NISTER_THRESHOLD)
        # decompose essential matrix
        R1, R2, t = decomposeEssentialMat(E)
        print("Rotation #1 = {0:s}".format(str(Rodrigues(R1)[0])))
        print("Rotation #2 = {0:s}".format(str(Rodrigues(R2)[0])))
        print("Translation = {0:s}".format(str(t)))
        return R1, R2, t

    def get_camera_motion_using_PnP(self, list_of_matches: List[DMatch],
                                    new_processed_frame_data: ProcessedFrameData):
        """
        This method estimates camera motion using the following steps:
        1. receive a set of 2D features, matched between the 1-st and the 2-nd frame
        2. estimate the 3D locations of the features, using their coordinates in the 1-st frame and the camera height
        3. solve the 2-nd frame camera pose, with the help of the PnP algorithm, by using the estimated 3D locations
            and their coordinates in the 2-nd frame
        """

        L: int = len(list_of_matches)
        array_of_3D_location: np.ndarray = np.empty((L, 3))
        array_of_2D_coords_first_frame: np.ndarray = np.empty((L, 2))
        array_of_2D_coords_second_frame: np.ndarray = np.empty((L, 2))
        H: float = self.robot_height_in_meter
        K_inv: np.ndarray = np.linalg.inv(self.camera_intrinsic)
        fx: float = self.camera_intrinsic[0, 0]
        fy: float = self.camera_intrinsic[1, 1]
        for match_idx, match in enumerate(list_of_matches):
            # estimate 3D location of feature, using 1-st frame coordinates
            first_frame_feature_coords = self.previous_frame_data.list_of_keypoints[match.queryIdx].pt
            second_frame_feature_coords = new_processed_frame_data.list_of_keypoints[match.trainIdx].pt
            # array_of_3D_location[match_idx, :] = H * np.array([first_frame_feature_coords[0] / fx,
            #                                                    first_frame_feature_coords[1] / fy,
            #                                                    1.0])
            array_of_3D_location[match_idx, :] = K_inv @ np.array([H * first_frame_feature_coords[0],
                                                                   H * first_frame_feature_coords[1],
                                                                   H])
            array_of_2D_coords_first_frame[match_idx, :] = np.array([first_frame_feature_coords[0],
                                                                     first_frame_feature_coords[1]])
            array_of_2D_coords_second_frame[match_idx, :] = np.array([second_frame_feature_coords[0],
                                                                      second_frame_feature_coords[1]])

        # check re-projection of 3D points
        re_project, _ = projectPoints(objectPoints=array_of_3D_location,
                                      rvec=np.zeros(3),
                                      tvec=np.zeros(3),
                                      cameraMatrix=self.camera_intrinsic,
                                      distCoeffs=None)
        reprojection_error = np.linalg.norm(x=re_project.squeeze() - array_of_2D_coords_first_frame,
                                            ord=2)

        retval, rvec, tvec, inliers = solvePnPRansac(objectPoints=array_of_3D_location,
                                                     imagePoints=array_of_2D_coords_second_frame,
                                                     cameraMatrix=self.camera_intrinsic,
                                                     distCoeffs=None)
        if retval:
            self.motion_estimation_status = MotionEstimationStatus.MOTION_OK
            self.current_robot_pose = RobotPose.build(position_x_meter=tvec[0][0],
                                                      position_y_meter=tvec[1][0],
                                                      yaw_angle_deg=radiansToDegrees * rvec[2][0])


if __name__ == "__main__":
    current_dir: str = os.getcwd()
    list_of_frame_paths: List[str] = list()
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_1.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_2.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_3.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_4.png'))
    list_of_frame_paths.append(os.path.join(current_dir, 'test/frame_5.png'))

    robot_height_in_meter = 2.5
    camera_intrinsic: np.ndarray = np.array([[1422.0, .0, 1024.0 / 2],
                                             [.0, 1422.0, 768.0 / 2],
                                             [.0, .0, 1.0]])
    robot_motion_estimator: RobotHorizontalMotionEstimator = RobotHorizontalMotionEstimator(
        robot_height_in_meter=robot_height_in_meter,
        camera_intrinsic=camera_intrinsic,
        debug_flag=True)

    for frame_path in list_of_frame_paths:
        frame: Optional[np.ndarray] = imread(frame_path)
        if frame is not None:
            robot_motion_estimator.set_new_frame(raw_frame=frame)
            robot_motion_estimator.print_current_position()
        else:
            print("Error! cannot read frame at location = {0:s}".format(frame_path))
            print("Aborting motion estimation")
            exit(1)
    waitKey(0)
    exit(0)
