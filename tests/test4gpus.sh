# in some cases launch gradio server, TGI server, or gradio server as inference server with +1 and +2 off base port
# So need to move gradio_server_port by mod 3 at least
GRADIO_SERVER_PORT=7860 TESTMODULOTOTAL=4 TESTMODULO=0 CUDA_VISIBLE_DEVICES=0 pytest -s -v -n 4 tests &> tests0.log &
GRADIO_SERVER_PORT=7870 TESTMODULOTOTAL=4 TESTMODULO=1 CUDA_VISIBLE_DEVICES=1 pytest -s -v -n 4 tests &> tests1.log &
GRADIO_SERVER_PORT=7880 TESTMODULOTOTAL=4 TESTMODULO=2 CUDA_VISIBLE_DEVICES=2 pytest -s -v -n 4 tests &> tests2.log &
GRADIO_SERVER_PORT=7890 TESTMODULOTOTAL=4 TESTMODULO=3 CUDA_VISIBLE_DEVICES=3 pytest -s -v -n 4 tests &> tests3.log &
