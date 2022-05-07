python3 inference.py --norm clean >> ./weight/clean_default.txt
python3 inference.py --task SSL --norm clean
python3 inference.py --norm l_1 >> ./weight/l_1_default.txt
python3 inference.py --task SSL --norm l_1 --attack_iter 5
python3 inference.py --task SSL --norm l_1 --attack_iter 10
python3 inference.py --task SSL --norm l_1 --attack_iter 15
python3 inference.py --norm l_2 >> ./weight/l_2_default.txt
python3 inference.py --task SSL --norm l_2 --attack_iter 5
python3 inference.py --task SSL --norm l_2 --attack_iter 10
python3 inference.py --task SSL --norm l_2 --attack_iter 15
python3 inference.py --norm l_inf >> ./weight/l_inf_default.txt
python3 inference.py --task SSL --norm l_inf --attack_iter 5
python3 inference.py --task SSL --norm l_inf --attack_iter 10
python3 inference.py --task SSL --norm l_inf --attack_iter 15

