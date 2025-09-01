from ImagesCameras import ImageTensor as im


color_path = '/home/godeta/PycharmProjects/TIR2VIS/frame_000000 (1).npy'
path = '/home/godeta/PycharmProjects/TIR2VIS/frame_000000.npy'

im(color_path).show(split_channel=True)
im(path).show(split_channel=True)