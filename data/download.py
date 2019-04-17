from subprocess import call
from data.tools import new_folder

patient_dat_file_src_root_path = 'box:/4Bek/Patient/'
healthy_dat_file_src_root_path = 'box:/4Bek/Healthy/'

patient_mcoe_file_src_root_path = 'box:/4Bek/Recons_Aug_2018_NoNorm/Y90/'
healthy_mcoe_file_src_root_path = 'box:/4Bek/Recons_Aug_2018_NoNorm/Healthy/'

dst_root_path = '/export/project/gan.weijie/dataset/mri_source/'

dat_file = [
    [
        'meas_MID00375_FID104201_CAPTURE_MoCo_RadialVIBE_CihatEldeniz',
        patient_dat_file_src_root_path + 'Y90_PMR37/',
        dst_root_path + 'patient_37/'],

    [
        'meas_MID00066_FID104789_CAPTURE_MoCo_RadialVIBE_CihatEldeniz',
        patient_dat_file_src_root_path + 'Y90_PMR38/',
        dst_root_path + 'patient_38/'],

    [
        'meas_MID00167_FID106710_CAPTURE_GA_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR40/',
        dst_root_path + 'patient_40/'],

    [
        'meas_MID00214_FID113048_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR44/',
        dst_root_path + 'patient_44/'],

    [
        'meas_MID00350_FID113184_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR45_HighRes/',
        dst_root_path + 'patient_45/'],

    [
        'meas_MID00087_FID117891_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR48/',
        dst_root_path + 'patient_48/'],

    [
        'meas_MID00051_FID118262_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR49/',
        dst_root_path + 'patient_49/'],

    [
        'meas_MID00105_FID118316_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR50/',
        dst_root_path + 'patient_50/'],

    [
        'meas_MID00148_FID119877_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR52/',
        dst_root_path + 'patient_52/'],

    [
        'meas_MID00158_FID121222_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR54/',
        dst_root_path + 'patient_54/'],

    [
        'meas_MID00197_FID121732_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR55/',
        dst_root_path + 'patient_55/'],

    # [
    #     'meas_MID00346_FID121881_CAPTURE_MOCO_Sag_Image',
    #     patient_dat_file_src_root_path + 'Y90_PMR56/',
    #     dst_root_path + 'patient_56/'],
    #  Have no 2000 scanlines?

    # [
    #     'meas_MID00227_FID126315_CAPTURE_MOCO_Sag_Image',
    #     patient_dat_file_src_root_path + 'Y90_PMR59/',
    #     dst_root_path + 'patient_59/'],
    # Have no .dat file...

    [
        'meas_MID00505_FID130498_CAPTURE_MOCO_Sag_Image',
        patient_dat_file_src_root_path + 'Y90_PMR60/',
        dst_root_path + 'patient_60/'],

    [
        'meas_MID00384_FID100412_fl3d_vibe_AM_ZeroPhi_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj01/',
        dst_root_path + 'healthy_01/'],

    [
        'meas_MID00405_FID100433_fl3d_vibe_AM_ZeroPhi_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj02/',
        dst_root_path + 'healthy_02/'],

    [
        'meas_MID00431_FID100459_fl3d_vibe_AM_ZeroPhi_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj03/',
        dst_root_path + 'healthy_03/'],

    [
        'meas_MID00631_FID100655_fl3d_vibe_AM_ZeroPhi_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj04/',
        dst_root_path + 'healthy_04/'],

    [
        'meas_MID00081_FID100745_fl3d_vibe_AM_ZeroPhi_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj05/',
        dst_root_path + 'healthy_05/'],

    [
        'meas_MID00118_FID105583_CAPTURE_GA_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj06/',
        dst_root_path + 'healthy_06/'],

    [
        'meas_MID00197_FID106044_CAPTURE_GA_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj07/',
        dst_root_path + 'healthy_07/'],

    [
        'meas_MID00220_FID106067_CAPTURE_GA_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj08/',
        dst_root_path + 'healthy_08/'],

    [
        'meas_MID00655_FID106502_CAPTURE_GA_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj09/',
        dst_root_path + 'healthy_09/'],

    [
        'meas_MID00301_FID114358_CAPTURE_GA_Sag_Image',
        healthy_dat_file_src_root_path + 'Subj10/',
        dst_root_path + 'healthy_10/'],
]

file_len = dat_file.__len__()

# END AT 15
for i in range(15, file_len):
    print("Downloading: [%d] file, totally [%d]" % (i + 1, file_len))
    new_folder(dat_file[i][2])
    call(['rclone', 'copy', dat_file[i][1] + dat_file[i][0] + '.dat', dat_file[i][2]])
    new_folder(dat_file[i][2] + dat_file[i][0] + '/')
    call(['rclone', 'copy', dat_file[i][1] + dat_file[i][0] + '/', dat_file[i][2] + dat_file[i][0] + '/'])
    exit(233)

