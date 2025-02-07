use nabla_ml::nab_math::NabMath;
use rand;
use plotters::prelude::*;
use nabla_ml::nab_array::NDArray;
use std::collections::HashMap;



fn plot_sigmoid(x_vec: &[f64], y_vec: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("sigmoid.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("f(sigmoid(x))", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-10.0..10.0, 0.0..1.0)?;

    chart.configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x_vec.iter().zip(y_vec.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("sigmoid(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_relu(x_vec: &[f64], z_vec: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("relu.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("f(ReLU(x))", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-10.0..10.0, 0.0..10.0)?; // Adjust the y-axis range as needed

    chart.configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x_vec.iter().zip(z_vec.iter()).map(|(&x, &z)| (x, z)),
            &BLUE,
        ))?
        .label("ReLU(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_leaky_relu(x_vec: &[f64], w_vec: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("leaky_relu.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("f(Leaky ReLU(x))", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-10.0..10.0, -1.0..10.0)?; // Adjust the y-axis range as needed

    chart.configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x_vec.iter().zip(w_vec.iter()).map(|(&x, &w)| (x, w)),
            &BLUE,
        ))?
        .label("Leaky ReLU(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_loss_history(epochs: &[f64], loss: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("loss_history.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Loss History (Linear Regression with MSE)", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..epochs.len() as f64,
            loss.iter().fold(f64::INFINITY, |a, &b| a.min(b))..loss[0]
        )?;

    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            epochs.iter().zip(loss.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

#[allow(non_snake_case)]
fn plot_regression(X: &[f64], y: &[f64], theta_0: f64, theta_x1: f64) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("regression_plot.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    // Find min and max X values for plotting the regression line
    let x_min = X.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = X.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Create points for regression line
    let line_x = vec![x_min, x_max];
    let line_y = line_x.iter().map(|&x| theta_0 + theta_x1 * x).collect::<Vec<f64>>();

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Linear Regression", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            x_min..x_max,
            y.iter().fold(f64::INFINITY, |a, &b| a.min(b))..
            y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        )?;

    chart.configure_mesh()
        .x_desc("X")
        .y_desc("y")
        .draw()?;

    // Plot scatter points using Circle
    chart
        .draw_series(
            X.iter().zip(y.iter()).map(|(&x, &y)| {
                Circle::new((x, y), 3, BLUE.filled())
            })
        )?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot regression line
    chart
        .draw_series(LineSeries::new(
            line_x.iter().zip(line_y.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Regression Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_training_history(history: &HashMap<String, Vec<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    // Plot accuracy
    let root_area = BitMapBackend::new("mnist_accuracy.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let epochs: Vec<f64> = (0..history["accuracy"].len()).map(|x| x as f64).collect();
    
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Model Accuracy", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..epochs.len() as f64,
            0f64..1f64
        )?;

    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;

    // Plot training accuracy
    chart
        .draw_series(LineSeries::new(
            epochs.iter().zip(history["accuracy"].iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Training")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot validation accuracy
    chart
        .draw_series(LineSeries::new(
            epochs.iter().zip(history["val_accuracy"].iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Validation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Plot loss
    let root_area = BitMapBackend::new("mnist_loss.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Model Loss", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..epochs.len() as f64,
            0f64..history["loss"].iter().fold(0f64, |a, &b| a.max(b))
        )?;

    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    // Plot training loss
    chart
        .draw_series(LineSeries::new(
            epochs.iter().zip(history["loss"].iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Training")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot validation loss
    chart
        .draw_series(LineSeries::new(
            epochs.iter().zip(history["val_loss"].iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Validation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    
    let n = 100;
    let x = NDArray::linspace(-10.00, 10.00, n, 1);
    let y = NabMath::sigmoid(&x);
    let z = NabMath::relu(&x);
    let w = NabMath::leaky_relu(&x, Some(0.01));

    // Convert NDArray to Vec<f64> for iteration
    let x_vec: Vec<f64> = (0..x.size()).map(|i| x.get(i)).collect();
    let y_vec: Vec<f64> = (0..y.size()).map(|i| y.get(i)).collect();
    let z_vec: Vec<f64> = (0..z.size()).map(|i| z.get(i)).collect();
    let w_vec: Vec<f64> = (0..w.size()).map(|i| w.get(i)).collect();

    // Generate a simple dataset
    let X = NDArray::from_vec((0..100).map(|_| 2.0 * rand::random::<f64>()).collect());
    let y = NDArray::from_vec(X.data().iter().map(|&x| 4.0 + 3.0 * x + rand::random::<f64>()).collect());

    // Apply linear regression
    let m = X.size() as f64;
    let mut theta_0 = 0.0;
    let mut theta_1 = 0.0;
    let learning_rate = 0.01;
    let num_epochs = 1000;
    let mut loss_history = Vec::with_capacity(num_epochs);
    
    // Training loop
    for epoch in 0..num_epochs {
        let predictions = X.clone() * theta_1 + theta_0;
        let errors = predictions.clone() - y.clone();
        
        // Convert errors to Vec<f64> if needed
        let errors_vec: Vec<f64> = (0..errors.size()).map(|i| errors.get(i)).collect();

        // Compute gradients
        let d_theta_0 = (1.0 / m) * errors_vec.iter().sum::<f64>();
        let d_theta_1 = (1.0 / m) * X.data().iter().zip(errors_vec.iter()).map(|(&x, &e)| x * e).sum::<f64>();
        
        // Update parameters
        theta_0 -= learning_rate * d_theta_0;
        theta_1 -= learning_rate * d_theta_1;
        
        // Compute and store loss
        let mse = (1.0 / (2.0 * m)) * errors_vec.iter().map(|e| e * e).sum::<f64>();
        loss_history.push(mse);
    }

    // Plot the results
    let epochs: Vec<f64> = (0..num_epochs).map(|x| x as f64).collect();
    if let Err(e) = plot_loss_history(&epochs, &loss_history) {
        eprintln!("Error plotting loss history: {}", e);
    }

    if let Err(e) = plot_regression(X.data(), y.data(), theta_0, theta_1) {
        eprintln!("Error plotting regression: {}", e);
    }

    // Plot activation functions
    if let Err(e) = plot_sigmoid(&x_vec, &y_vec) {
        eprintln!("Error plotting sigmoid: {}", e);
    }
    if let Err(e) = plot_relu(&x_vec, &z_vec) {
        eprintln!("Error plotting relu: {}", e);
    }
    if let Err(e) = plot_leaky_relu(&x_vec, &w_vec) {
        eprintln!("Error plotting leaky relu: {}", e);
    }

    println!("Final parameters: theta_0 = {}, theta_1 = {}", theta_0, theta_1);


    // Work on the mnist dataset
    // Access the data
    // let ((mut train_images, train_labels), (mut test_images, test_labels)) = 
    // NDArray::load_and_split_dataset("datasets/mnist", 80.0)?;

    // println!("Training samples: {:?}", train_images.shape());
    // println!("Test samples: {:?}", test_images.shape());

    // // Extract and print the 42nd entry
    // println!("Label of 42nd entry: {}", train_labels.get(42));
    // println!("Image of 42nd entry:");
    // let image_42: NDArray = train_images.extract_sample(42);
    // image_42.pretty_print(0);
   
    // train_images.normalize();
    // test_images.normalize();

    // // Reshape to (num_images, 784) using -1 to infer the number of images
    // let reshaped_images = train_images.reshape(vec![-1, 784]);
    // let reshaped_test_images = test_images.reshape(vec![-1, 784]);
    // println!("Reshaped images shape: {:?}", reshaped_images.shape());
  
    // let image_42_2d: NDArray = reshaped_images.extract_sample(42);
    // image_42_2d.pretty_print(2);

    // let one_hot_train_labels: NDArray = NDArray::one_hot_encode(&train_labels);
    // let one_hot_test_labels: NDArray = NDArray::one_hot_encode(&test_labels);

    // println!("One hot test labels shape: {:?}", one_hot_test_labels.shape());

    // // print the first 10 one hot encoded labels
    // let label_42: NDArray =  one_hot_train_labels.extract_sample(42);
    // label_42.pretty_print(1);


    // let mut model = Model::new()
    //     .input(vec![784])  // Input layer: 784 features (28x28 pixels)
    //     .add_dense(32, Box::new(Sigmoid::default()))  // Larger first hidden layer
    //     .add_dense(32, Box::new(Sigmoid::default()))   // Second hidden layer
    //     .add_dense(10, Box::new(Softmax::default()))   // Output layer: 10 classes
    //     .build();


    // // Customize the optimizer with a specific learning rate
    // let optimizer = Adam::new(0.0001, 0.9, 0.999, 1e-8);

    // model.compile(
    //     Box::new(optimizer),
    //     Box::new(CategoricalCrossentropy),
    //     vec!["accuracy".to_string()]
    // );

    // // Add validation data to monitor training
    // let history = model.fit(
    //     &reshaped_images, 
    //     &one_hot_train_labels,
    //     64,  // batch size
    //     10,  // epochs
    // );


    // println!("Training History:");
    // for (epoch, metrics) in history.iter().enumerate() {
    //     println!("Epoch {}: {:?}", epoch + 1, metrics);
    // }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use nabla_ml::{nab_io::{load_nab, loadz_nab, save_nab, savez_nab}, nab_utils::NabUtils};
    use nabla_ml::nab_model::NabModel;
    use nabla_ml::nab_layers::NabLayer;
    use nabla_ml::nab_model::reset_node_id;
    #[test]
    fn test_array_save_and_load() -> Result<(), Box<dyn std::error::Error>> {
        let array = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        save_nab("data.nab", &array)?;

        let loaded_array = load_nab("data.nab")?;
        assert_eq!(array.data(), loaded_array.data());
        assert_eq!(array.shape(), loaded_array.shape());

        // Clean up the test file
        // std::fs::remove_file("data.nab")?;
        Ok(())
    }

    #[test]
    fn test_multi_save_and_load() -> Result<(), Box<dyn std::error::Error>> {
        let array1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let array2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);

        // Save arrays with specified names
        savez_nab("data1.nab", vec![("x", &array1), ("y", &array2)])?;

        // Load multiple arrays - Changed filename to match the save operation
        let arrays: HashMap<String, NDArray> = loadz_nab("data1.nab")?;
        
        // Access individual arrays by name
        let x = arrays.get("x").unwrap();
        let y = arrays.get("y").unwrap();

        assert_eq!(array1.data(), x.data());
        assert_eq!(array2.data(), y.data());

        // Clean up test file
        std::fs::remove_file("data1.nab")?;

        Ok(())
    }

    #[test]
    fn test_load_and_split_dataset() -> Result<(), Box<dyn std::error::Error>> {
        // Load and split the dataset
        let ((train_images, train_labels), (test_images, test_labels)) = 
            NabUtils::load_and_split_dataset("./datasets/mnist", 80.0)?;

        // Access the data
        println!("Training samples: {}", train_images.shape()[0]);
        println!("Test samples: {}", test_images.shape()[0]);

        // Add assertions to verify the dataset is loaded and split correctly
        assert!(train_images.shape()[0] > 0, "Training samples should be greater than 0");
        assert!(test_images.shape()[0] > 0, "Test samples should be greater than 0");

        // Ensure labels are also correctly loaded
        assert!(train_labels.shape()[0] > 0, "Training labels should be greater than 0");
        assert!(test_labels.shape()[0] > 0, "Test labels should be greater than 0");

        Ok(())
    }

    #[test]
    fn test_minst() -> Result<(), Box<dyn std::error::Error>> {

        // 1. Load the dataset

        let ((x_train, y_train), (x_test, y_test)) = NabUtils::load_and_split_dataset("datasets/mnist", 80.0).unwrap();

        // Step 2: Normalize input data (scale pixels to 0-1)
        println!("Normalizing data...");
        let x_train = x_train.divide_scalar(255.0);
        let x_test = x_test.divide_scalar(255.0);

        // Step 3: Reshape input data
        let x_train = x_train.reshape(&[x_train.shape()[0], 784])
            .expect("Failed to reshape training data");
        let x_test = x_test.reshape(&[x_test.shape()[0], 784])
            .expect("Failed to reshape test data");

        // Step 3: One-hot encode target data
        println!("One-hot encoding targets...");
        let y_train = NDArray::one_hot_encode(&y_train);
        let y_test = NDArray::one_hot_encode(&y_test);

        println!("Data shapes:");
        println!("x_train: {:?}", x_train.shape());
        println!("y_train: {:?}", y_train.shape());
        println!("x_test: {:?}", x_test.shape());
        println!("y_test: {:?}", y_test.shape());

        // Step 3: Create model architecture
        println!("Creating model...");

        // Reset node ID counter before test
        reset_node_id();
        
        // Create a simple model
        let input = NabModel::input(vec![784]);
        let dense1 = NabLayer::dense(784, 32, Some("relu"), Some("dense1"));
        let x = input.apply(dense1);

        let dense2 = NabLayer::dense(32, 32, Some("relu"), Some("dense2"));
        let x = x.apply(dense2);

        let output_layer = NabLayer::dense(32, 10, Some("softmax"), Some("output"));
        let output = x.apply(output_layer);
        
        let mut model = NabModel::new_functional(vec![input], vec![output]);
       
        model.summary();

        model.compile(
            "sgd",                      
            0.1,                        
            "categorical_crossentropy", 
            vec!["accuracy".to_string()]
        );

    
        // Step 5: Train model
        println!("Training model...");
        let history = model.fit(
            &x_train,
            &y_train,
            32,             // Increase batch size from 32 to 64
            10,             // Increase epochs from 2 to 10
            Some((&x_test, &y_test))
        );

        model.save_compressed("mnist_model.ez").unwrap();

        let mut model_loaded = NabModel::load_compressed("mnist_model.ez").unwrap();

        // Step 6: Evaluate final model
        println!("Evaluating model...");
        let eval_metrics = model_loaded.evaluate(&x_test, &y_test, 32);
        
        // Print final results
        println!("Final test accuracy: {:.2}%", eval_metrics["accuracy"] * 100.0);

        // Step 7: Verify model achieved reasonable accuracy (>85%)
        assert!(eval_metrics["accuracy"] > 0.85, 
            "Model accuracy ({:.2}%) below expected threshold", 
            eval_metrics["accuracy"] * 100.0
        );

        // Verify training history contains expected metrics
        assert!(history.contains_key("loss"));
        assert!(history.contains_key("accuracy"));
        assert!(history.contains_key("val_loss"));
        assert!(history.contains_key("val_accuracy"));

        // After training the model
        plot_training_history(&history)?;

        Ok(())
    }

    
}


