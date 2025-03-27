from cpipe.tools.cpipetools import CPipeTools

# The encryption model in cpipe format can be used under all CPIPE frameworks.
# CPipeTools.encrypt_models("../../__OTHERS__/demo_person/movenet_person_pose.onnx", model_type=CPipeTools.MODEL_TYPE_CPIPE)


# The codex encrypted model can only be used on the specified license device.
# CPipeTools.encrypt_models("../../__OTHERS__/demo_person/movenet_person_pose.onnx",
#                           license_password="1234567890123456",
#                           license_path="../../__OTHERS__/demo_person/1234567890123456.cpipe.license",
#                           model_type=CPipeTools.MODEL_TYPE_CODEX)

CPipeTools.encrypt_models("../../src/model_files/416x416-det_10g_batch.onnx",
                          model_type=CPipeTools.MODEL_TYPE_CPIPE)
