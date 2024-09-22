import moviepy.editor as mpy
import os


main_folder = "demo_images"       
sub_folder = "demo_images/edited"     
output_video = "_output_video.mp4"    

duration = 2

video_width = 1280
video_height = 720

image_files = [f for f in os.listdir(main_folder) if f.endswith(('.jpg', '.png'))]

clips = []

for image_file in image_files:
    main_image_path = os.path.join(main_folder, image_file)
    sub_image_path = os.path.join(sub_folder, f"modified_{image_file}")  # La imagen en la subcarpeta

    
    if os.path.exists(sub_image_path):
        main_image = mpy.ImageClip(main_image_path).set_duration(duration)
        
        sub_image = mpy.ImageClip(sub_image_path).set_duration(duration)

        clips.append(main_image)
        
        # main_image = main_image.resize(width=video_width // 2)  
        # sub_image = sub_image.resize(width=video_width // 2)   

        combined = mpy.clips_array([[main_image, sub_image]]).set_duration(duration)

        clips.append(combined)

final_video = mpy.concatenate_videoclips(clips, method = "compose")
final_video.write_videofile(output_video, fps=24, codec="libx264", preset="slow", bitrate="5000k", threads=4, audio=False)


