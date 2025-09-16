/*****************************************
 *Create an Interactive perceptron in C
 *****************************************/

#include <stdio.h>

// Step function (activation function)
int step_function(int weighted_sum) 
{
    return (weighted_sum >= 5) ? 1 : 0;  // Hiring threshold is 5
}

// Perceptron function
int perceptron(int experience, int skills, float weights[], float bias) 
{
    int weighted_sum = (experience * weights[0]) + (skills * weights[1]) + bias;
    return step_function(weighted_sum);
}

// Trainable Perceptron Learning Algorithm
void train_perceptron(  int training_data[][2], 
                        int expected_output[], 
                        float weights[], float *bias, int num_samples, float learning_rate, int epochs) 
{
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        int total_error = 0;

        for (int i = 0; i < num_samples; i++) 
        {
            int experience = training_data[i][0];
            int skills = training_data[i][1];

            int prediction = perceptron(experience, skills, weights, *bias);
            int error = expected_output[i] - prediction;
            total_error += error * error;  // Sum of squared errors

            // Adjust weights using the Perceptron Learning Rule
            weights[0] += learning_rate * error * experience;
            weights[1] += learning_rate * error * skills;
            *bias += learning_rate * error;
        }

        // Stop if the perceptron has learned correctly
        if (total_error == 0) 
        {
            printf("Training completed in %d epochs.\n", epoch + 1);
            break;
        }
    }
}

int main() 
{
    // Training data: (Experience, Skills) -> Expected Output (Hire/Reject)
    int training_data[3][2] = 
    {
        {2, 2},  // Experience = 2, Skills = 2 (Should be hired)
        {1, 1},  // Experience = 1, Skills = 1 (Should be rejected)
        {3, 1}   // Experience = 3, Skills = 1 (Should be hired)
    };
    
    int expected_output[3] = {1, 0, 1}; // Expected hiring decisions

    // Initialize weights and bias randomly
    float weights[2] = {0, 0};  // Start with no knowledge
    float bias = 0;
    
    float learning_rate = 0.1; // Small adjustments each time
    int epochs = 100;  // Maximum training iterations

    // Train the perceptron
    train_perceptron(training_data, expected_output, weights, &bias, 3, learning_rate, epochs);

    // Display the final weights
    printf("Final Weights: Experience = %f, Skills = %f, Bias = %f\n", weights[0], weights[1], bias);

    // Interactive mode: Allow user to input new candidates
    while (1) 
    {
        int experience, skills;
        printf("\nEnter candidate's experience and skills (or -1 to exit): ");
        scanf("%d %d", &experience, &skills);

        if (experience == -1 || skills == -1) 
        {
            printf("Exiting...\n");
            break;
        }

        int prediction = perceptron(experience, skills, weights, bias);
        if (prediction == 1) 
        {
            printf("Prediction: Hired!\n");
        } 
        else 
        {
            printf("Prediction: Rejected\n");
        }
    }

    return 0;
}
/**************************************************************************/
