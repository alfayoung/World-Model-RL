from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
#Consistent with the web interface, the value and citic rewards of video input can be evaluated.


#assign local model path
model_path="/local_data/cf3331/vlac_dsrl/VLAC/checkpoint"
#download model from https://huggingface.co/InternRobotics/VLAC

#assign video path and task description
# use python switch case
task_id = 33
match task_id:
    case 57:
        test_video='/local_data/cf3331/dsrl_pi0/logs/DSRL_pi0_DoubleQ_Libero/dsrl_double_q_libero_2025_12_12_22_10_10_0000--s-0/real_trajectory_videos/step_0001860/real_ep_34_seed_[1269942167 1896434130]_q_-1.861_ret_1.00_success.mp4'
        ref_video='/local_data/cf3331/dsrl_pi0/logs/DSRL_pi0_DoubleQ_Libero/dsrl_double_q_libero_2025_12_12_22_10_10_0000--s-0/real_trajectory_videos/step_0000000/real_ep_2_seed_[1510749119 2670142958]_q_0.297_ret_1.00_success.mp4'
        task_description = 'Pick up the cream cheese and put it in the tray.'
    case 33:
        test_video='/local_data/cf3331/dsrl_pi0/logs/DSRL_pi0_DoubleQ_Libero/dsrl_double_q_libero_2025_12_11_10_27_08_0000--s-0/real_trajectory_videos/step_0003780/real_ep_35_seed_[2344706688 1827681567]_q_-4.287_ret_1.00_success.mp4'
        # ref_video=None
        ref_video='/local_data/cf3331/vlac_dsrl/ref_img/KITCHEN_SCENE6_close_the_microwave/KITCHEN_SCENE6_close_the_microwave-demo_0.mp4'
        task_description = 'Close the microwave.'
    case 45:
        test_video='/local_data/cf3331/dsrl_pi0/logs/DSRL_pi0_Libero/dsrl_pi0_libero_2025_12_16_05_59_21_0000--s-0/trajectory_videos/step_0000000/ep_6_ret_0.00_fail.mp4'
        ref_video='/local_data/cf3331/vlac_dsrl/KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it/KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it-demo_0.mp4'
        # ref_video=None
        task_description = 'Turn on the stove and put the frying pan on it.'

#init model
Critic=GAC_model(tag='critic')
Critic.init_model(model_path=model_path,model_type='internvl2',device_map='cuda:0')
Critic.temperature=0.5
Critic.top_k=1
Critic.set_config()
Critic.set_system_prompt()

# transform video
test_video_compressed = os.path.join(os.path.dirname(test_video),"test.mp4")
_,output_fps=compress_video(test_video, test_video_compressed,fps=5)
reference_video_compressed = None
if ref_video:
    reference_video_compressed = os.path.join(os.path.dirname(ref_video),"ref.mp4")
    compress_video(ref_video, reference_video_compressed,fps=5)


# generate Critic results
result_path,value_list,critic_list,done_list = Critic.web_trajectory_critic(
    task_description=task_description,
    main_video_path=test_video_compressed,
    reference_video_path=reference_video_compressed,#if None means no reference video, only use task_description to indicate the task
    batch_num=5,#batch number
    ref_num=6,#image number used in reference video
    think=False,# whether to CoT
    skip=5,#pair-wise step
    rich=False,#whether to output decimal value
    reverse_eval=False,#whether to reverse the evaluation(for VROC evaluation)
    output_path="results",
    fps=float(output_fps),
    frame_skip=True,#whether to skip frames(if false, each frame while be evaluated, cost more time)
    done_flag=False,#whether to out put done value
    in_context_done=False,#whether use reference video to generate done value
    done_threshold=0.9,#done threshold
    video_output=True#whether to output video
)


print("=" * 100)
print(">>>>>>>>>Critic results<<<<<<<<<<")
print(" ")

print(f"result path: {result_path}")
print(f"task description: {task_description}")
print("=" * 50)

print("value_list:")
print(value_list)
print("=" * 50)

print("critic_list:")
print(critic_list)
print("=" * 50)

print("done_list:")
print(done_list)
print("=" * 100)