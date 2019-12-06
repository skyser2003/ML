import math
import random
import time
import itertools

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import util.helper as helper
from util.plot_utils import Plot_Reproduce_Performance
from cq.cq_data import CqData, CqDataType, CqDataMode


class DCGan:
    def __init__(self):
        pass

    def loss_gp_disc(self, dataloader, gen, disc, num_batch: int, num_z: int, num_cat: int, criterion,
                     wgan_grad_output):
        disc.zero_grad()

        # Real & fake x
        batch_data = torch.tensor(dataloader.get_batch(num_batch, True), dtype=torch.float32)

        z_val, cat_input = generate_z_val(num_batch, num_z, num_cat)
        x_gen = gen(z_val)

        # Disc
        disc_real, _ = disc(batch_data)
        disc_fake, cat_output = disc(x_gen)

        # Improved WGAN
        eps = torch.rand((num_batch, 1)).expand(batch_data.size())
        scale_fn = 10

        x_pn = eps * batch_data + (1 - eps) * x_gen
        disc_pn, _ = disc(x_pn)

        grad = \
            autograd.grad(disc_pn, x_pn, grad_outputs=wgan_grad_output, create_graph=True,
                          retain_graph=True)
        grad = grad[0]
        grad = grad.norm(dim=1)

        ddx = scale_fn * (grad - 1) ** 2
        ddx = ddx.mean()

        loss_real = (disc_real - disc_fake).mean() + ddx

        if num_cat != 0:
            loss_cat = cat_loss_calc(criterion, cat_output, cat_input)
            loss_real += loss_cat

        loss_real.backward()

        return disc_real.mean(), loss_real

    def loss_gp_gen(self, gen, disc, num_batch: int, num_z: int, num_cat: int, criterion):
        gen.zero_grad()

        z_val, cat_input = generate_z_val(num_batch, num_z, num_cat)
        x_gen = gen(z_val)
        disc_fake, cat_output = disc(x_gen)

        disc_fake = disc_fake.mean()

        loss_fake = disc_fake

        if num_cat != 0:
            loss_cat = cat_loss_calc(criterion, cat_output, cat_input)
            loss_fake += loss_cat

        loss_fake.backward()

        return disc_fake, loss_fake

    def loss_ls_disc(self, dataloader, gen, disc, num_batch: int, num_z: int, num_cat: int, criterion):
        disc.zero_grad()

        # Real & fake x
        batch_data = torch.tensor(dataloader.get_batch(num_batch, True), dtype=torch.float32)
        z_val, cat_input = generate_z_val(num_batch, num_z, num_cat)
        x_gen = gen(z_val)

        disc_real, _ = disc(batch_data)
        disc_fake, cat_output = disc(x_gen)

        loss_real = 0.5 * (((disc_real - 1) ** 2).mean() + (disc_fake ** 2).mean())

        if num_cat != 0:
            loss_cat = cat_loss_calc(criterion, cat_output, cat_input)
            loss_real += loss_cat

        disc_real = disc_real.mean()
        loss_real.backward()

        return disc_real, loss_real

    def loss_ls_gen(self, gen, disc, num_batch: int, num_z: int, num_cat: int):
        gen.zero_grad()

        # Real & fake x
        z_val, cat_input = generate_z_val(num_batch, num_z, num_cat)
        x_gen = gen(z_val)

        disc_fake, cat_output = disc(x_gen)

        loss_fake = 0.5 * ((disc_fake - 1) ** 2).mean()

        disc_fake = disc_fake.mean()

        loss_fake.backward()

        return disc_fake, loss_fake

    def run(self):
        dataloader = CqData(CqDataType.FACE, max_data=64, mode=CqDataMode.BG_BLACK | CqDataMode.GRAY_SCALE,
                            scale_down=3,
                            nchw=True)

        image_width = dataloader.get_image_width()
        image_height = dataloader.get_image_height()
        num_sample_x = 8
        num_sample_y = 8
        num_batch = num_sample_x * num_sample_y

        output_dir = "pytorch/output"
        helper.clean_create_dir(output_dir)

        writer = SummaryWriter("pytorch/log")
        saver = Plot_Reproduce_Performance(output_dir, num_sample_x, num_sample_y, image_width, image_height, 3,
                                           nchw=True)

        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cudnn.benchmark = True

        num_cat = 0
        num_z = 256 - num_cat
        num_image_channel = dataloader.get_channel_count()
        lr = 0.0002

        gen = Generator(num_z, num_cat, image_width, image_height, num_image_channel)
        disc = Discriminator(image_width, image_height, num_cat, num_image_channel)
        disc_with_gen = DiscWithGen(gen, disc)

        wgan_grad_output = torch.ones((num_batch, 1 + num_cat))

        criterion = nn.CrossEntropyLoss()

        opt_gen = optim.Adam(gen.parameters(), lr)
        opt_disc = optim.Adam(disc.parameters(), lr)

        num_step_d = 3

        print_i = 0
        num_print = int(1e2)
        num_summary = 10

        # Tensorboard graph
        sample_z_val, _ = generate_z_val(num_batch, num_z, num_cat)
        writer.add_graph(disc_with_gen, sample_z_val, verbose=True)

        fixed_cat_input = None
        if num_cat != 0:
            fixed_cat_input = torch.zeros([num_batch, num_cat])

            for i in range(num_batch):
                row = fixed_cat_input[i]
                if i < row.size(0):
                    row[i] = 1

        torch.cuda.synchronize()
        begin = time.time()

        for total_i in itertools.count():
            print_debug = total_i % num_print == 0 and total_i != 0

            with autograd.profiler.profile(enabled=print_debug, use_cuda=True) as prof:
                for param in disc.parameters():
                    param.requires_grad_(True)

                for _ in range(num_step_d):
                    # disc_real, loss_real = self.loss_gp_disc(dataloader, gen, disc, num_batch, num_z, num_cat, criterion, wgan_grad_output)
                    disc_real, loss_real = self.loss_ls_disc(dataloader, gen, disc, num_batch, num_z, num_cat,
                                                             criterion)

                    opt_disc.step()

                # Generator train
                for param in disc.parameters():
                    param.requires_grad_(False)

                # disc_fake, loss_fake = self.loss_gp_gen(gen, disc, num_batch, num_z, num_cat, criterion)
                disc_fake, loss_fake = self.loss_ls_gen(gen, disc, num_batch, num_z, num_cat)

                opt_gen.step()

            # For debugging
            if total_i % num_summary == 0:
                writer.add_scalar("Summary/loss_fake", loss_fake, total_i)
                writer.add_scalar("Summary/loss_real", loss_real, total_i)
                writer.add_scalar("Summary/disc_fake", disc_fake, total_i)
                writer.add_scalar("Summary/disc_real", disc_real, total_i)

                if num_cat != 0:
                    writer.add_scalar("Summary/loss_cat", loss_cat, total_i)

            if print_debug:
                torch.cuda.synchronize()
                now = time.time()
                debug_message = "%.1fk steps done, %.5f seconds elapsed - " % (total_i / 1000, now - begin)
                debug_message += "Disc G : %f, Disc D : %f, Loss G: %f, Loss D: %f" % (
                    disc_fake, disc_real, loss_fake, loss_real)

                if num_cat != 0:
                    debug_message += ", Loss cat: %f" % loss_cat

                print(debug_message)

                prof.export_chrome_trace(f"trace_output{print_i}.json")

                with torch.no_grad():
                    output_z_val, _ = generate_z_val(num_batch, num_z, 0)
                    if num_cat != 0:
                        output_z_val = concat_z_val(output_z_val, fixed_cat_input)

                    fake = gen(output_z_val).detach().cpu().numpy()

                filename = f"output{print_i}.png"
                saver.save_pngs(fake, num_image_channel, filename)

                print_i += 1
                begin = now

        print("All done")


class Generator(nn.Module):
    def __init__(self, num_z: int, num_cat: int, width: int, height: int, num_image_channel: int):
        super(Generator, self).__init__()

        initial_multiplier = 0.5

        self.num_initial_channel = num_image_channel * 32
        self.initial_width = int(width * initial_multiplier)
        self.initial_height = int(height * initial_multiplier)

        num_conv = 4
        kernel_size = 5
        stride_size = 2
        padding_size = 2
        output_padding_size = 1

        num_in_channel = self.num_initial_channel
        num_out_channel = num_in_channel // 2

        self.linear = nn.Sequential(
            nn.Linear(num_z + num_cat, self.num_initial_channel * self.initial_width * self.initial_height),
            nn.LeakyReLU(inplace=True)
        )

        conv = []

        for i in range(num_conv - 1):
            conv.extend([
                nn.ConvTranspose2d(num_in_channel, num_out_channel, kernel_size, stride_size, padding_size,
                                   output_padding_size),
                nn.LeakyReLU(inplace=True)
            ])

            num_in_channel = num_out_channel
            num_out_channel = num_in_channel // 2

        avg_pool_kernel_size = int(initial_multiplier * stride_size ** num_conv)

        conv.extend([
            nn.ConvTranspose2d(num_in_channel
                               , num_image_channel, kernel_size, stride_size, padding_size,
                               output_padding_size),
            nn.AvgPool2d(avg_pool_kernel_size, avg_pool_kernel_size),
            nn.Sigmoid()
        ])

        for layer in conv:
            if type(layer) == nn.ConvTranspose2d or type(layer) == nn.Linear:
                nn.init.normal_(layer.weight, 0, 0.02)
                nn.init.constant_(layer.bias, 0)

        self.main = nn.Sequential(*conv)

    def forward(self, input):
        linear = self.linear(input)
        linear_image = linear.view(-1, self.num_initial_channel, self.initial_height, self.initial_width)

        return self.main(linear_image).view(linear.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, width: int, height: int, num_cat: int, num_image_channel: int):
        super(Discriminator, self).__init__()

        self.width = width
        self.height = height
        self.num_image_channel = num_image_channel
        self.num_cat = num_cat

        num_conv = 4
        kernel_size = 5
        stride_size = 2
        padding_size = 2
        output_width = width
        output_height = height

        num_in_channel = num_image_channel
        num_out_channel = num_image_channel * 32

        output = []

        for i in range(num_conv - 1):
            output.extend([
                nn.Conv2d(num_in_channel, num_out_channel, kernel_size, stride_size, padding_size),
                nn.LeakyReLU(inplace=True)
            ])

            output_width = calc_conv2d_output_size(output_width, kernel_size, stride_size, padding_size, 1)
            output_height = calc_conv2d_output_size(output_height, kernel_size, stride_size, padding_size, 1)

            num_in_channel = num_out_channel
            num_out_channel = num_in_channel * 2

        output.append(nn.Conv2d(num_in_channel, 1 + num_cat, (output_width, output_height)))

        for layer in output:
            if type(layer) == nn.Conv2d:
                nn.init.normal_(layer.weight, 0, 0.02)
                nn.init.constant_(layer.bias, 0)

        self.main = nn.Sequential(*output)

    def forward(self, input: torch.Tensor):
        input = input.view(-1, self.num_image_channel, self.height, self.width)
        result = self.main(input).view(input.size(0), -1)
        disc_val, cat_output = result.split([1, self.num_cat], dim=1)
        cat_output = cat_output.detach()

        return disc_val, cat_output


class DiscWithGen(nn.Module):
    def __init__(self, gen: Generator, disc: Discriminator):
        super(DiscWithGen, self).__init__()

        self.gen = gen
        self.disc = disc

    def forward(self, input: torch.Tensor):
        x_gen = self.gen(input)
        output = self.disc(x_gen)

        return output


def generate_z_val(num_batch: int, num_z: int, num_cat: int):
    real_z = torch.randn((num_batch, num_z))

    cat_input = None
    if num_cat != 0:
        cat_input = generate_random_cat(num_batch, num_cat)

    z_val = concat_z_val(real_z, cat_input)
    return z_val, cat_input


def concat_z_val(real_z: torch.Tensor, cat_input: torch.Tensor):
    if cat_input is not None:
        return torch.cat([real_z, cat_input.to(dtype=torch.float32)], dim=1)
    else:
        return real_z


def generate_random_cat(batch_size: int, num_cat: int):
    cat_arr = torch.zeros([batch_size, num_cat], dtype=torch.long)

    for i in range(batch_size):
        index = random.randint(0, num_cat - 1)
        cat_arr[i][index] = 1

    return cat_arr


def cat_loss_calc(criterion: torch.nn.CrossEntropyLoss, output: torch.Tensor, target: torch.Tensor):
    if target is None:
        return None

    results = torch.zeros(output.size(0))

    for i in range(output.size(0)):
        target_row = target[i]
        output_row = output[i].view(-1, 1)

        dummy_row = torch.zeros(output_row.size())
        output_row = torch.cat([dummy_row, output_row], dim=1)

        result = criterion(output_row, target_row)
        results[i] = result

    return results.mean()


def calc_conv2d_output_size(length: int, kernel: int, stride: int, padding: int, dilation: int):
    return math.floor((length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def calc_trans_conv2d_output_size(length: int, stride: int, padding: int, kernel_size: int, output_padding: int):
    return (length - 1) * stride - 2 * padding + kernel_size + output_padding


if __name__ == "__main__":
    sisr = DCGan()
    sisr.run()
