import math
import torch
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import Pool

class PatchManager():
    def __init__(self, img, patch_size, scale, receptive_field, model, device='cpu'):
        self.scale = scale
        self.receptive_field = receptive_field
        self.img = img
        self.image_h = img.size(2)
        self.image_w = img.size(3)
        self.patch_size = patch_size
        self.overlap = receptive_field - scale if receptive_field % scale == 0 else scale * math.floor(receptive_field / scale)
        assert self.patch_size >= 2* self.overlap , f"x_patch_size ({self.patch_size}) should be larger than twice the overlap ({self.overlap})"
        self.y_patch_size = self.patch_size // self.scale
        self.y_h = self.image_h // self.scale
        self.y_w = self.image_w // self.scale
        self.x_step_size = self.patch_size - self.overlap
        self.y_step_size = self.x_step_size // self.scale
        self.y_remove_size = self.overlap // (2 * self.scale)
        self.x_remove_size = self.y_remove_size * self.scale
        self.h_block = math.ceil((self.image_h - self.patch_size) / self.x_step_size) + 1
        self.w_block = math.ceil((self.image_w - self.patch_size) / self.x_step_size) + 1
        self.edge_h_method = True if (self.image_h - self.patch_size) % self.x_step_size != 0 else False
        self.edge_w_method = True if (self.image_w - self.patch_size) % self.x_step_size != 0 else False
        self.device = device
        self.model = model.to(self.device)
    
    def encode_patch(self, args):
        i, j = args
        x_patch = self.img[:, :, i : i + self.patch_size, j : j + self.patch_size]
        out = self.model.g_a(x_patch)
        return out
    
    def decode_patch(self, args):
        i, j = args
        y_patch = self.y_hat[:, :, i : i + self.y_patch_size, j : j + self.y_patch_size]
        out = self.model.g_s(y_patch)
        return out
    
    def encode_bigimage(self):
        out = self.model.g_a(self.img)
        return out
    
    def decode_bigimage(self, y_hat):
        out = self.model.g_s(y_hat)
        return out
    
    def get_encode_grid(self):
        image_grid_start = [[[0, 0] for j in range(self.w_block)] for i in range(self.h_block)]
        feature_grid_mode = [[0 for j in range(self.w_block)] for i in range(self.h_block)]
        for i in range(self.h_block):
            for j in range(self.w_block):
                # get image domain grid start
                if not self.edge_h_method:
                    image_grid_start[i][j][0] = i*self.x_step_size
                else:
                    if i == self.h_block - 1:
                        image_grid_start[i][j][0] = self.image_h - self.patch_size
                    else:
                        image_grid_start[i][j][0] = i*self.x_step_size
                if not self.edge_w_method:
                    image_grid_start[i][j][1] = j*self.x_step_size
                else:
                    if j == self.w_block - 1:
                        image_grid_start[i][j][1] = self.image_w - self.patch_size
                    else:
                        image_grid_start[i][j][1] = j*self.x_step_size
                """
                # get feature domain grid mode
                    0##4##1
                    #######
                    5##8##7
                    #######
                    2##6##3
                """
                if i == 0 and j == 0:
                    feature_grid_mode[i][j] = 0
                elif i == 0 and j == self.w_block - 1:
                    feature_grid_mode[i][j] = 1
                elif i == self.h_block - 1 and j == 0:
                    feature_grid_mode[i][j] = 2
                elif i == self.h_block - 1 and j == self.w_block - 1:
                    feature_grid_mode[i][j] = 3
                elif i == 0:
                    feature_grid_mode[i][j] = 4
                elif j == 0:
                    feature_grid_mode[i][j] = 5
                elif i == self.h_block - 1:
                    feature_grid_mode[i][j] = 6
                elif j == self.w_block - 1:
                    feature_grid_mode[i][j] = 7
                else:
                    feature_grid_mode[i][j] = 8
        return image_grid_start, feature_grid_mode
    
    def get_decode_grid(self):
        feature_grid_start = [[[0, 0] for j in range(self.w_block)] for i in range(self.h_block)]
        image_grid_mode = [[0 for j in range(self.w_block)] for i in range(self.h_block)]

        for i in range(self.h_block):
            for j in range(self.w_block):
                # get image domain grid start
                if not self.edge_h_method:
                    feature_grid_start[i][j][0] = i*self.y_step_size
                else:
                    if i == self.h_block - 1:
                        feature_grid_start[i][j][0] = self.y_h - self.y_patch_size
                    else:
                        feature_grid_start[i][j][0] = i*self.y_step_size
                if not self.edge_w_method:
                    feature_grid_start[i][j][1] = j*self.y_step_size
                else:
                    if j == self.w_block - 1:
                        feature_grid_start[i][j][1] = self.y_w - self.y_patch_size
                    else:
                        feature_grid_start[i][j][1] = j*self.y_step_size
                """
                # get image domain grid mode
                    0##4##1
                    #######
                    5##8##7
                    #######
                    2##6##3
                """
                if i == 0 and j == 0:
                    image_grid_mode[i][j] = 0
                elif i == 0 and j == self.w_block - 1:
                    image_grid_mode[i][j] = 1
                elif i == self.h_block - 1 and j == 0:
                    image_grid_mode[i][j] = 2
                elif i == self.h_block - 1 and j == self.w_block - 1:
                    image_grid_mode[i][j] = 3
                elif i == 0:
                    image_grid_mode[i][j] = 4
                elif j == 0:
                    image_grid_mode[i][j] = 5
                elif i == self.h_block - 1:
                    image_grid_mode[i][j] = 6
                elif j == self.w_block - 1:
                    image_grid_mode[i][j] = 7
                else:
                    image_grid_mode[i][j] = 8
        return feature_grid_start, image_grid_mode
    
    def concat_feature_buffer(self, buffer, mode):
        meta_y = torch.zeros(1, self.model.N, self.y_h, self.y_w).to(self.device)
        val1 = self.y_patch_size - self.y_remove_size
        index =  0
        if self.edge_h_method:
            y_h_edge = self.y_h - (val1 + (self.h_block - 2) * self.y_step_size)
        if self.edge_w_method:
            y_w_edge = self.y_w - (val1 + (self.w_block - 2) * self.y_step_size)
        for i in range(self.h_block):
            for j in range(self.w_block):
                if mode[i][j] == 0:
                    meta_y[:, :, :val1, :val1] = buffer[index][:,:,:-self.y_remove_size,:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 1:
                    if self.edge_w_method:
                        meta_y[:, :, :val1, -y_w_edge:] = buffer[index][:,:,:-self.y_remove_size,-y_w_edge:] if self.y_remove_size > 0 else buffer[index][:,:,:,-y_w_edge:]
                    else:
                        meta_y[:, :, :val1, -val1:] = buffer[index][:,:,:-self.y_remove_size,self.y_remove_size:] if self.y_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 2:
                    if self.edge_h_method:
                        meta_y[:, :, -y_h_edge:, :val1] = buffer[index][:,:,-y_h_edge:,:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,-y_h_edge:,:]
                    else:
                        meta_y[:,:,-val1:,:val1] = buffer[index][:,:,self.y_remove_size:,:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,:]
                elif mode[i][j] == 3:
                    if self.edge_h_method and self.edge_w_method:
                        meta_y[:, :, -y_h_edge:, -y_w_edge:] = buffer[index][:,:,-y_h_edge:,-y_w_edge:]
                    elif self.edge_w_method:
                        meta_y[:, :, -val1:, -y_w_edge:] = buffer[index][:,:, self.y_remove_size:, -y_w_edge:]
                    elif self.edge_h_method:
                        meta_y[:, :, -y_h_edge:, -val1:] = buffer[index][:,:,-y_h_edge:,self.y_remove_size:]
                    else:
                        meta_y[:, :, -val1:, -val1:] = buffer[index][:,:,self.y_remove_size:,self.y_remove_size:]
                elif mode[i][j] == 4:
                    meta_y[:, :, :val1, val1+(j-1)*self.y_step_size:val1+j*self.y_step_size] = buffer[index][:,:,:-self.y_remove_size,self.y_remove_size:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 5:
                    meta_y[:, :, val1+(i-1)*self.y_step_size:val1+i*self.y_step_size, :val1] = buffer[index][:,:,self.y_remove_size:-self.y_remove_size,:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,:]
                elif mode[i][j] == 6:
                    if self.edge_h_method:
                        meta_y[:, :, -y_h_edge:, val1+(j-1)*self.y_step_size:val1+j*self.y_step_size] = buffer[index][:,:,-y_h_edge:,self.y_remove_size:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,-y_h_edge:,self.y_remove_size:]
                    else:
                        meta_y[:, :, -val1:, val1+(j-1)*self.y_step_size:val1+j*self.y_step_size] = buffer[index][:,:,self.y_remove_size:,self.y_remove_size:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,self.y_remove_size:]
                elif mode[i][j] == 7:
                    if self.edge_w_method:
                        meta_y[:, :, val1+(i-1)*self.y_step_size:val1+i*self.y_step_size, -y_w_edge:] = buffer[index][:,:,self.y_remove_size:-self.y_remove_size,-y_w_edge:] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,-y_w_edge:]
                    else:
                        meta_y[:, :, val1+(i-1)*self.y_step_size:val1+i*self.y_step_size, -val1:] = buffer[index][:,:,self.y_remove_size:-self.y_remove_size,self.y_remove_size:] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,self.y_remove_size:]
                else:
                    meta_y[:, :, val1+(i-1)*self.y_step_size:val1+i*self.y_step_size, val1+(j-1)*self.y_step_size:val1+j*self.y_step_size] = buffer[index][:,:,self.y_remove_size:-self.y_remove_size,self.y_remove_size:-self.y_remove_size] if self.y_remove_size > 0 else buffer[index][:,:,self.y_remove_size:,self.y_remove_size:]
                index += 1
        return meta_y
    
    def concat_image_buffer(self, buffer, mode):
        meta_x = torch.zeros(1, 3, self.image_h, self.image_w).to(self.device)
        val1 = self.patch_size - self.x_remove_size
        index =  0
        if self.edge_h_method:
            x_h_edge = self.image_h - (val1 + (self.h_block - 2) * self.x_step_size)
        if self.edge_w_method:
            x_w_edge = self.image_w - (val1 + (self.w_block - 2) * self.x_step_size)
        for i in range(self.h_block):
            for j in range(self.w_block):
                if mode[i][j] == 0:
                    meta_x[:, :, :val1, :val1] = buffer[index][:,:,:-self.x_remove_size,:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 1:
                    if self.edge_w_method:
                        meta_x[:, :, :val1, -x_w_edge:] = buffer[index][:,:,:-self.x_remove_size,-x_w_edge:] if self.x_remove_size > 0 else buffer[index][:,:,:,-x_w_edge:]
                    else:
                        meta_x[:, :, :val1, -val1:] = buffer[index][:,:,:-self.x_remove_size,self.x_remove_size:] if self.x_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 2:
                    if self.edge_h_method:
                        meta_x[:, :, -x_h_edge:, :val1] = buffer[index][:,:,-x_h_edge:,:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,-x_h_edge:,:]
                    else:
                        meta_x[:,:,-val1:,:val1] = buffer[index][:,:,self.x_remove_size:,:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,:]
                elif mode[i][j] == 3:
                    if self.edge_h_method and self.edge_w_method:
                        meta_x[:, :, -x_h_edge:, -x_w_edge:] = buffer[index][:,:,-x_h_edge:,-x_w_edge:]
                    elif self.edge_w_method:
                        meta_x[:, :, -val1:, -x_w_edge:] = buffer[index][:,:, self.x_remove_size:, -x_w_edge:]
                    elif self.edge_h_method:
                        meta_x[:, :, -x_h_edge:, -val1:] = buffer[index][:,:,-x_h_edge:,self.x_remove_size:]
                    else:
                        meta_x[:, :, -val1:, -val1:] = buffer[index][:,:,self.x_remove_size:,self.x_remove_size:]
                elif mode[i][j] == 4:
                    meta_x[:, :, :val1, val1+(j-1)*self.x_step_size:val1+j*self.x_step_size] = buffer[index][:,:,:-self.x_remove_size,self.x_remove_size:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,:,:]
                elif mode[i][j] == 5:
                    meta_x[:, :, val1+(i-1)*self.x_step_size:val1+i*self.x_step_size, :val1] = buffer[index][:,:,self.x_remove_size:-self.x_remove_size,:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,:]
                elif mode[i][j] == 6:
                    if self.edge_h_method:
                        meta_x[:, :, -x_h_edge:, val1+(j-1)*self.x_step_size:val1+j*self.x_step_size] = buffer[index][:,:,-x_h_edge:,self.x_remove_size:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,-x_h_edge:,self.x_remove_size:]
                    else:
                        meta_x[:, :, -val1:, val1+(j-1)*self.x_step_size:val1+j*self.x_step_size] = buffer[index][:,:,self.x_remove_size:,self.x_remove_size:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,self.x_remove_size:]
                elif mode[i][j] == 7:
                    if self.edge_w_method:
                        meta_x[:, :, val1+(i-1)*self.x_step_size:val1+i*self.x_step_size, -x_w_edge:] = buffer[index][:,:,self.x_remove_size:-self.x_remove_size,-x_w_edge:] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,-x_w_edge:]
                    else:
                        meta_x[:, :, val1+(i-1)*self.x_step_size:val1+i*self.x_step_size, -val1:] = buffer[index][:,:,self.x_remove_size:-self.x_remove_size,self.x_remove_size:] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,self.x_remove_size:]
                else:
                    meta_x[:, :, val1+(i-1)*self.x_step_size:val1+i*self.x_step_size, val1+(j-1)*self.x_step_size:val1+j*self.x_step_size] = buffer[index][:,:,self.x_remove_size:-self.x_remove_size,self.x_remove_size:-self.x_remove_size] if self.x_remove_size > 0 else buffer[index][:,:,self.x_remove_size:,self.x_remove_size:]
                index += 1
        return meta_x


    def encode(self):
        image_grid_start, feature_grid_mode = self.get_encode_grid()
        buffer = []
        for i in range(self.h_block):
            for j in range(self.w_block):
                buffer.append(self.encode_patch((image_grid_start[i][j][0], image_grid_start[i][j][1])))
        meta_y = self.concat_feature_buffer(buffer, feature_grid_mode)
        return meta_y

    def encode_cpu_multi_threading(self, threadings=30):
        image_grid_start, feature_grid_mode = self.get_encode_grid()
        with ThreadPoolExecutor(max_workers=threadings) as executor:
            futures = [
                executor.submit(self.encode_patch, (image_grid_start[i][j][0], image_grid_start[i][j][1]))
                for i in range(self.h_block) for j in range(self.w_block)
            ]
        wait(futures)
        buffer = [future.result() for future in futures]
        meta_y = self.concat_feature_buffer(buffer, feature_grid_mode)
        return meta_y
    
    def encode_cpu_multi_processing(self, threadings=30):
        image_grid_start, feature_grid_mode = self.get_encode_grid()
        with Pool(threadings) as pool:
            buffer = pool.map(self.encode_patch, [(image_grid_start[i][j][0], image_grid_start[i][j][1]) for i in range(self.h_block) for j in range(self.w_block)])
        meta_y = self.concat_feature_buffer(buffer, feature_grid_mode)
        return meta_y
    
    def encode_batch_processing(self, batch_size):
        image_grid_start, feature_grid_mode = self.get_encode_grid()
        buffer = []
        buffer_y = []
        for i in range(self.h_block):
            for j in range(self.w_block):
                buffer.append(self.img[:, :, image_grid_start[i][j][0] : image_grid_start[i][j][0] + self.patch_size, image_grid_start[i][j][1] : image_grid_start[i][j][1] + self.patch_size])
                if len(buffer) == batch_size:
                    patches = torch.cat(buffer, dim=0) 
                    patches_y = self.model.g_a(patches) 
                    buffer_y += list(torch.split(patches_y, 1, dim=0))  # 切分成单个patch的结果
                    buffer = []
                    
        if len(buffer) > 0:
            patches = torch.cat(buffer, dim=0)
            patches_y = self.model.g_a(patches)
            buffer_y += list(torch.split(patches_y, 1, dim=0))
            
        meta_y = self.concat_feature_buffer(buffer_y, feature_grid_mode)
        return meta_y
    
    def decode(self, y_hat):
        self.y_hat = y_hat
        feature_grid_start, image_grid_mode = self.get_decode_grid()
        buffer = []
        for i in range(self.h_block):
            for j in range(self.w_block):
                buffer.append(self.decode_patch((feature_grid_start[i][j][0], feature_grid_start[i][j][1])))
        meta_x = self.concat_image_buffer(buffer, image_grid_mode)
        return meta_x
    
    def decode_cpu_multi_threading(self, y_hat, threadings=30):
        self.y_hat = y_hat
        feature_grid_start, image_grid_mode = self.get_decode_grid()
        with ThreadPoolExecutor(max_workers=threadings) as executor:
            futures = [
                executor.submit(self.decode_patch, (feature_grid_start[i][j][0], feature_grid_start[i][j][1]))
                for i in range(self.h_block) for j in range(self.w_block)
            ]
        wait(futures)
        buffer = [future.result() for future in futures]
        meta_x = self.concat_image_buffer(buffer, image_grid_mode)
        return meta_x
    
    def decode_cpu_multi_processing(self, y_hat, threadings=30):
        self.y_hat = y_hat
        feature_grid_start, image_grid_mode = self.get_decode_grid()
        with Pool(threadings) as pool:
            buffer = pool.map(self.decode_patch, [(feature_grid_start[i][j][0], feature_grid_start[i][j][1]) for i in range(self.h_block) for j in range(self.w_block)])
        meta_x = self.concat_image_buffer(buffer, image_grid_mode)
        return meta_x
    
    def decode_batch_processing(self, y_hat, batch_size):
        self.y_hat = y_hat
        feature_grid_start, image_grid_mode = self.get_decode_grid()
        buffer = []
        buffer_x = []
        for i in range(self.h_block):
            for j in range(self.w_block):
                buffer.append(self.y_hat[:, :, feature_grid_start[i][j][0] : feature_grid_start[i][j][0] + self.y_patch_size, feature_grid_start[i][j][1] : feature_grid_start[i][j][1] + self.y_patch_size])
                if len(buffer) == batch_size:
                    patches = torch.cat(buffer, dim=0) 
                    patches_x = self.model.g_s(patches) 
                    buffer_x += list(torch.split(patches_x, 1, dim=0))  # 切分成单个patch的结果
                    buffer = []
        
        if len(buffer) > 0:
            patches = torch.cat(buffer, dim=0)
            patches_x = self.model.g_s(patches)
            buffer_x += list(torch.split(patches_x, 1, dim=0))
        
        meta_x = self.concat_image_buffer(buffer_x, image_grid_mode)
        return meta_x