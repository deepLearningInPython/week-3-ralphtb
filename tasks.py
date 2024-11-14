import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    output_length = input_length - kernel_length + 1
    return output_length



# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Step 1: Calculate the length of the output array using the function from Task 1.
    output_length = compute_output_size_1d(input_array, kernel_array)
    
    # Step 2: Initialize an empty output array of the calculated length.
    output_array = np.zeros(output_length)
    
    # Step 3: Perform the convolution using a loop.
    # Slide the kernel over the input array and compute the sum of element-wise multiplication.
    for i in range(output_length):
        # Get the segment of the input that the kernel will multiply.
        segment = input_array[i : i + len(kernel_array)]
        
        # Perform element-wise multiplication and sum the result.
        output_array[i] = np.sum(segment * kernel_array)
    
    # Return the convolution result.
    return output_array


# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # Get the height and width of the input matrix
    input_height, input_width = input_matrix.shape
    
    # Get the height and width of the kernel matrix
    kernel_height, kernel_width = kernel_matrix.shape
    
    # Calculate the output height and width using the formula
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    # Return the dimensions as a tuple
    return (output_height, output_width)


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Step 1: Calculate the output dimensions using the function from Task 3.
    output_height, output_width = compute_output_size_2d(input_matrix, kernel_matrix)
    
    # Step 2: Initialize an empty output matrix with zeros.
    output_matrix = np.zeros((output_height, output_width))
    
    # Step 3: Get the dimensions of the kernel matrix.
    kernel_height, kernel_width = kernel_matrix.shape
    
    # Step 4: Perform the convolution using nested loops.
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current region of the input matrix that the kernel will overlap.
            region = input_matrix[i:i + kernel_height, j:j + kernel_width]
            
            # Perform element-wise multiplication and sum the result.
            output_matrix[i, j] = np.sum(region * kernel_matrix)
    
    # Return the resulting output matrix.
    return output_matrix


# -----------------------------------------------