from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
				"https://rich-text-to-image.github.io/image_assets/color_tri.png",	# str (filepath or URL to image) in 'Upload' Image component
				"anything",	# str in 'Detection Prompt[To detect multiple objects, seperating each name with '.', like this: cat . dog . chair ]' Textbox component
				"remove",	# str in 'Task type' Radio component
				"anything",	# str in 'Inpaint Prompt (if this is empty, then remove)' Textbox component
				0,	# int | float (numeric value between 0.0 and 1.0) in 'Box Threshold' Slider component
				0,	# int | float (numeric value between 0.0 and 1.0) in 'Text Threshold' Slider component
				0,	# int | float (numeric value between 0.0 and 1.0) in 'IOU Threshold' Slider component
				"merge",	# str in 'inpaint_mode' Radio component
				"type what to detect below",	# str in 'Mask from' Radio component
				"segment",	# str in 'remove mode' Radio component
				"10",	# str in 'remove_mask_extend' Textbox component
				3,	# int | float (numeric value between 1 and 20) in 'How many relations do you want to see' Slider component
				"Detailed",	# str in 'Kosmos Description Type' Radio component
				fn_index=2
)
print(result)