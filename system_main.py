import gradio as gr
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import time
from testOEIS_Interface2 import return_oeis_formula


def image_id():
    time_now = time.time()
    id = int(round(time_now * 10000))
    return id


def formula_image(real_seq, pred_seq, formula, error_rate, id):
    # print("pred_seq:",pred_seq)
    x = []
    for i in range(len(real_seq)):
        x.append(i + 1)
        real_seq[i] = int(real_seq[i])

    pred_seq = pred_seq[:len(real_seq)]
    x = np.array(x)
    real_seq = np.array(real_seq)
    pred_seq = np.array(pred_seq)

    # print("x:", x)
    # print("real_seq:", real_seq)
    # print("pred_seq:", pred_seq)

    x_new = np.linspace(x.min(), x.max(), 80)  # 300 represents number of points to make between T.min and T.max
    # print("x_new:",x_new)
    y_smooth = make_interp_spline(x, real_seq)(x_new)
    # 散点图
    plt.scatter(x, pred_seq, c='red', s=65)  # alpha:透明度) c:颜色
    # 折线图
    # plt.plot(x, y, linewidth=1)  # 线宽linewidth=1
    # 平滑后的折线图
    plt.plot(x_new, y_smooth, c='black', linewidth=2.0)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # plt.title(f"序号为{id}的公式产生的数列图示", fontsize=16)  # 标题及字号
    plt.xlabel("n", fontdict={'family': 'Times New Roman', 'size': 16})  # X轴标题及字号
    plt.ylabel("a(n)", fontdict={'family': 'Times New Roman', 'size': 16})  # Y轴标题及字号
    plt.tick_params(axis='both', labelsize=14)  # 刻度大小
    # plt.xlim(x_x)
    # plt.xticks(x)
    # plt.axis([0, 1100, 1, 1100000])#设置坐标轴的取值范围
    image_dic = "image_save"
    if not os.path.exists(image_dic):
        os.makedirs(image_dic)
    image_path = f"{image_dic}/{image_id()}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    # plt.savefig(image_path, dpi=300)
    # plt.show()
    plt.close()
    return image_path


def formula_images_list(real_seq, formula_dic):
    show_image_lis = []
    for error_rate, formula_lis in formula_dic.items():
        id, formula, pred_seq = formula_lis
        image_path = formula_image(real_seq, pred_seq, formula, error_rate, id)
        show_image_lis.append(image_path)
    return show_image_lis


def greet(input_sequence, beam_size, show_formulas_nums, pred_terms_nums):
    finale_res = return_oeis_formula(input_sequence, beam_size, show_formulas_nums, pred_terms_nums)
    input_sequence = input_sequence.replace(" ", "")
    input_sequence = input_sequence.replace("、", ",")
    input_sequence_lis = input_sequence.split(',')
    seq_len = len(input_sequence_lis)
    seq_len = min(25, seq_len)
    print("")

    finale_res = finale_res[:show_formulas_nums]

    image_dic = {}

    pred_formulas = "|$$\\textbf{序号}$$| $$\\textbf{候选公式}$$ | $$\\textbf{预测输入数列的后" + str(
        pred_terms_nums) + "项}$$ | $$\\textbf{误差率}$$ | $$\\textbf{OEIS序号}$$ |\n| :----: | :----: | :----: | :----:  |  :----: |\n"

    for i, tuple in enumerate(finale_res):
        if tuple[4] != '--':
            seq_name_link = f"[{tuple[4]}](https://oeis.org/{tuple[4]})"
        else:
            seq_name_link = "--"
        pre_seq_str = ', '.join(str(num) for num in tuple[3][seq_len:])
        pred_formulas = pred_formulas + f"| $${i + 1}$$ |  $$ \large {str(tuple[2])}$$   | <center> <font size=3>{pre_seq_str}</font> | $${tuple[0]}\\\\%$$ | <font size=3> {seq_name_link} </font>  |\n"

        if str(tuple[0]) not in image_dic and len(image_dic) < 5:
            image_dic[str(tuple[0])] = [i + 1, str(tuple[2]), tuple[3]]
    #
    # print("%"*100)
    # print("markDown:", pred_formulas)
    # print("%"*100)

    show_image_lis = formula_images_list(input_sequence_lis, image_dic)
    # show_image_lis = ['D:\PythonWorkspace\\fairseq-main\\test2.png', 'D:\PythonWorkspace\\fairseq-main\\test2.png',
    #                   'D:\PythonWorkspace\\fairseq-main\\test2.png']
    return pred_formulas, show_image_lis


with gr.Blocks(css="style.css") as demo:
    gr.Markdown("""
        # <center>  OEIS整数数列公式发现系统 </center>
    """)
    # 设置输入组件 r

    with gr.Row():
        with gr.Column(scale=11):
            input_sequence = gr.Textbox(label="请输入一条整数数列, 用英文逗号分隔整数, 最佳输入项数为25项",
                                        value='0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610')
            with gr.Row():
                beam_size = gr.Slider(0, 100, value=32, label="束搜索大小", )
                show_formulas_nums = gr.Slider(0, 100, value=32, label="候选公式数量")
                pred_terms_nums = gr.Slider(0, 100, value=10, label="预测的项数")

        with gr.Column(scale=1, min_width=1):
            reset_btn = gr.ClearButton(value='重置', scale=1, size='lg').add([input_sequence])

        with gr.Column(scale=4):
            generate_btn = gr.Button("生成", scale=1, variant='primary', size='lg')
    with gr.Row():
        with gr.Column(scale=15):
            formulas_md = gr.Markdown(elem_classes='center')
        with gr.Column(scale=7, min_width=1):
            image_res_lis = gr.Gallery(min_width=1, columns=2)
            # 设置按钮点击事件
    generate_btn.click(fn=greet, inputs=[input_sequence, beam_size, show_formulas_nums, pred_terms_nums],
                       outputs=[formulas_md, image_res_lis], )

demo.launch(server_name='10.1.11.214', server_port=1580, inbrowser=True)
