<img width="1248" height="1248" alt="image" src="https://github.com/user-attachments/assets/acf1fa47-b97a-42d9-b6d7-c7260942e2fe" /># Yolov8_vs_Efficientdet_in_ship_detection
L’Intelligence Artificielle (IA) et le Deep Learning ont révolutionné la vision par ordinateur, permettant aux machines de comprendre et d’analyser des images avec une précision remarquable. Parmi ces avancées, la détection d’objets joue un rôle essentiel pour identifier et localiser automatiquement des éléments présents dans une scène.
# Introduction
Autonomous vehicles are becoming more common in various industries, but the use of autonomous maritime vehicles is still being studied. This is because controlling these vehicles requires making important decisions about design, propulsion, payload management, and communication systems, which can lead to errors and collisions. One major challenge is detecting other ships and objects in real-time to avoid collisions. Recently, deep learning techniques based on convolutional neural networks (CNNs) have been developed to help with this, such as YOLOv8 and EfficientDet. This paper examines how these methods can be used to detect ships. We trained and tested these two models on a large maritime dataset. On examining the performance of the two models we have compared the working of both.

# Motivation
- revolutionize waterway technology with AI, sensors, and autonomy.

- Growing demand for offshore exploration and surveillance fuels AMV usage.

- AMVs offer safety, cost, and efficiency advantages over traditional vessels.

- Applications of AMVs include research, monitoring, security, and more.

- Object detection remains a challenge for AMVs, especially in maritime settings.

- Deep learning techniques like CNNs improve object detection accuracy and speed.

- YOLOv8 and EfficientDet offer enhanced accuracy, reduced complexity, scalability, robustness, and generalization for ship detection.

# Literature Review:
Research on AMVs is ongoing and focuses on enabling autonomous vessels to operate in different marine environments, perform various tasks, and serve different applications.

Recent studies have explored AMV applications in oceanographic research, environmental monitoring, marine transportation, and defense. 

The current research aims to improve reliability, safety, and energy efficiency of AMVs, while developing new sensor technologies and data processing methods to enhance performance and functionality.

# Research Gaps:
- Improved accuracy in challenging maritime environments.
- Real-time operation on autonomous vehicles.
- Generalizability to different object types.
- Robustness to changes in lighting and background.
- Integration of sensor fusion techniques.
- Enhanced detection of small and distant objects.
- Evaluation of algorithms in real-world maritime scenarios.
- Consideration of regulatory and ethical aspects.
<img width="2250" height="1500" alt="image" src="https://github.com/user-attachments/assets/edd115e6-b3ed-4de5-bf12-e0f479877500" />
</n>
<img width="2250" height="1500" alt="image" src="https://github.com/user-attachments/assets/37b933d7-951c-4077-81b5-0b94d9fbd1dd" />
</n>
<img width="2250" height="1500" alt="image" src="https://github.com/user-attachments/assets/c4a67c56-5b67-4d2e-8888-e4a113769ce0" />
</n>
  # Objectives:
- To evaluate the performance of two state-of-the-art deep learning algorithms, YOLOv8 and EfficientDet, for object detection in autonomous maritime vehicles
-  To compare the performance of these algorithms on a real-world dataset of maritime objects and evaluate their accuracy, computational efficiency, and generalizability

  # Results:
A Python script specifying the thresholds, network paths and images to recognize wass used to test the new model. 
Our trained model correctly recognises the item (ship) in images and videos
The model was able to recognise the item not only in photos, but also in real time/inputted videos.
</n>
<img width="1248" height="1248" alt="image" src="https://github.com/user-attachments/assets/d709b5eb-9d0c-4efa-972f-c652ec2c8c15" />
</n>
<img width="1248" height="1248" alt="image" src="https://github.com/user-attachments/assets/2f08c6c3-98d0-41c9-a8cf-83a78e06881a" />
</n>
<img width="505" height="580" alt="image" src="https://github.com/user-attachments/assets/0c41b807-f76a-4f68-a13b-fac4f2b2160b" />
</n>
<img width="1248" height="1248" alt="image" src="https://github.com/user-attachments/assets/d431f616-5794-4743-b83f-85be8305613e" />
</n>

## Final Result of our research:
-in Results folder
-Ship Detection in Video
-Ship Detection in Images
### Datasets used :
- Singapore Maritime Dataset for Testing the model
- Roboflow Ship2 Image Dataset for Training the model
  
# References:
[1] Prasad, D.K., Rajan, D., Rachmawati, L., Rajabally, E. and Quek, C., 2017. Video processing from electro-optical sensors for object detection and tracking in a maritime environment: A survey. IEEE Transactions on Intelligent Transportation Sys- tems, 18(8), pp.1993-2016.

[2] Kim, T.E., Perera, L.P., Sollid, M.P., Batalden, B.M. and Sydnes, A.K., 2022. Safety challenges related to autonomous ships in mixed navigational environments. WMU Journal of Maritime Affairs, 21(2), pp.141-159.

[3] Shao, Z., Lyu, H., Yin, Y., Cheng, T., Gao, X., Zhang, W., Jing, Q., Zhao, Y. and Zhang, L., 2022. Multi-Scale Object Detection Model for Autonomous Ship Navigation in Maritime Environment. Journal of Marine Science and Engineering, 10(11), p.1783.

[4] Chang, Y.L., Anagaw, A., Chang, L., Wang, Y.C., Hsiao, C.Y. and Lee, W.H., 2019. Ship detection based on YOLOv2 for SAR imagery. Remote Sensing, 11(7), p.786.

[5] Munteanu, D., Moina, D., Zamfir, C.G., Petrea, S, M., Cristea, D.S. and Munteanu, N., 2022. Sea Mine Detection Framework Using YOLO, SSD and EfficientDet Deep Learning Models. Sensors, 22(23), p.9536.

[6] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

[7] smart helmet, Ship2 Dataset
