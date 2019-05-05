from ModelBuilder import ModelBuilder

modelBuilder = ModelBuilder()


training_data_location="./softwareeng/data/fer2013/fer2013.csv"
modelBuilder.generate_model(training_data_location)

modelBuilder.train_model("random")

directory = "./softwareeng/data"
facial_expression_json_name = "facial_expression_model_structure"
facial_experssion_weights_file_name = "facial_expression_model_weights"

modelBuilder.save_model(directory,facial_expression_json_name,facial_experssion_weights_file_name)