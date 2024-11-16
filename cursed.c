#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define W1x 10
#define W2x 10

double uniform_random(double max, double min)
{
    return min + (rand() / (double)RAND_MAX) * (max - min);
}
int **readcsv(const char *file, int *row, int *column)
{
    FILE *data = fopen(file, "r");
    if (!data)
    {
        perror("invalid File\n");
        return NULL;
    }
    char line[4000];
    int rowcount = 0;
    int columncount = 0;

    while (fgets(line, sizeof(line), data))
    {
        rowcount++;
        if (rowcount == 1)
        {
            char *token = strtok(line, ",");
            while (token)
            {
                columncount++;
                token = strtok(NULL, ",");
            }
        }
    }
    int **matrix = malloc(rowcount * sizeof(int *));
    for (int i = 0; i < rowcount; i++)
    {
        matrix[i] = malloc(columncount * sizeof(int));
    }
    rewind(data);

    int i = 0;
    int j = 0;
    while (fgets(line, sizeof(line), data))
    {
        char *token = strtok(line, ",");
        j = 0;
        while (token)
        {
            matrix[i][j++] = atoi(token);
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(data);
    *row = rowcount;
    *column = columncount;

    return matrix;
}

double **create_matrix(int row, int column)
{
    double **new_matrix = malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
    {
        new_matrix[i] = malloc(column * sizeof(double));
    }
    return new_matrix;
}

double **dot(int sizex1, int sizey1, int sizex2, int sizey2, double **mat1, double **mat2)
{
    if (sizey1 != sizex2)
    {
        printf("Invalid Multipication \n");
        return NULL;
    }

    double **sum = create_matrix(sizex1, sizey2);

    for (int i = 0; i < sizex1; i++)
    {
        for (int j = 0; j < sizey2; j++)
        {
            sum[i][j] = 0;
            for (int k = 0; k < sizey1; k++)
            {
                sum[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return sum;
}
void free_mat(int **mat, int sizex1)
{
    if (mat == NULL)
        return;
    for (int i = 0; i < sizex1; i++)
    {
        free(mat[i]);
        // printf("%d\n", i);
    }
    free(mat);
}
void free_mat_double(double **mat, int sizex1)
{
    if (mat == NULL)
        return;
    for (int i = 0; i < sizex1; i++)
    {
        free(mat[i]);
        // printf("%d\n", i);
    }
    free(mat);
}

double **transpose(int *row, int *column, int **mat)
{
    int sizex = *row;
    int sizey = *column;

    int **transposed = malloc(sizey * sizeof(double *));
    if (!transposed)
    {
        perror("failed to initialize\n");
        return NULL;
    }
    for (int i = 0; i < sizey; i++)
    {
        transposed[i] = malloc(sizex * sizeof(double));
    }

    for (int i = 0; i < sizex; i++)
    {
        for (int j = 0; j < sizey; j++)
        {
            transposed[j][i] = mat[i][j];
        }
    }
    int temp = *row;
    *row = *column;
    *column = temp;
    return transposed;
}

double **transpose_wo_change(int row, int column, int **mat)
{
    int sizex = row;
    int sizey = column;

    int **transposed = malloc(sizey * sizeof(double *));
    if (!transposed)
    {
        perror("failed to initialize\n");
        return NULL;
    }
    for (int i = 0; i < sizey; i++)
    {
        transposed[i] = malloc(sizex * sizeof(double));
    }

    for (int i = 0; i < sizex; i++)
    {
        for (int j = 0; j < sizey; j++)
        {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

void printMatrix(double **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

double **assign_values(int **mat, int startx, int finishx, int starty, int finishy)
{
    int rows = finishx - startx;
    int cols = finishy - starty;

    double **new_matrix = malloc(rows * sizeof(double *));
    if (!new_matrix)
    {
        perror("Failed to allocate memory for new_matrix\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++)
    {
        new_matrix[i] = malloc(cols * sizeof(double));
        if (!new_matrix[i])
        {
            perror("Failed to allocate memory for new_matrix\n");
            for (int k = 0; k < i; k++)
            {
                free(new_matrix[k]);
            }
            free(new_matrix);
            return NULL;
        }
        for (int j = 0; j < cols; j++)
        {
            new_matrix[i][j] = mat[i + startx][j + starty];
        }
    }
    return new_matrix;
}

double **Xavier_init(int row, int column)
{
    double placeholder = row + column;
    double limit = sqrt(6.0 / placeholder);

    double **mat = create_matrix(row, column);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            mat[i][j] = uniform_random(limit, -limit);
        }
    }
    return mat;
}

double **zeros(int row, int col)
{
    double **mat = create_matrix(row, col);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            mat[i][j] = 0;
        }
    }
    return mat;
}

double **Leaky_relu(double **mat, int row, int column)
{
    double **output = create_matrix(row, column);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            output[i][j] = max(mat[i][j], 0.001 * mat[i][j]);
        }
    }
    return output;
}

double **softmax(double **mat, int row, int column)
{
    double **output = create_matrix(row, column);
    double sum = 0;
    for (int j = 0; j < column; j++)
    {
        sum = 0;
        double max_val = mat[0][j];
        for (int i = 0; i < row; i++)
        {
            if (mat[i][j] > max_val)
            {
                max_val = mat[i][j];
            }
        }
        for (int i = 0; i < row; i++)
        {
            output[i][j] = exp(mat[i][j] - max_val);
            sum += output[i][j];
        }
        for (int i = 0; i < row; i++)
        {
            output[i][j] = output[i][j] / sum;
        }
    }
    return output;
}
double **boradcast(int sizex1, int sizey1, int sizex2, int sizey2, double **mat1, double **mat2)
{
    if (sizex1 != sizex2 || sizey2 != 1)
    {
        perror("Invalid addition");
        return NULL;
    }
    double **mat = create_matrix(sizex1, sizey1);
    for (int i = 0; i < sizex1; i++)
    {
        for (int j = 0; j < sizey1; j++)
        {
            mat[i][j] = mat1[i][j] + mat2[i][0];
        }
    }
    return mat;
}
// W1, b1,W2,b2,W3, b3, X

double **leaky_relu_deriv(double **mat, int row, int column)
{
    double **new = create_matrix(row, column);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            if (mat[i][j] <= 0)
            {
                new[i][j] = 0.001;
            }
            else
            {
                new[i][j] = 1;
            }
        }
    }
    return new;
}

double **One_Hot(double **mat, int row, int column)
{
    double **ohy = malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
    {
        ohy[i] = calloc(column, sizeof(double));
    }

    int class_index;
    for (int i = 0; i < row; i++)
    {
        class_index = mat[i][0];
        if (class_index >= 0 && class_index < column)
        {
            ohy[i][class_index] = 1.0;
        }
    }
    return ohy;
}
double **matrix_arithmetic(double **mat1, double **mat2, int row, int column, char op)
{
    double **result = create_matrix(row, column);
    if (op == 'p')
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
    }
    else if (op == 'm')
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
    }
    return result;
}

int main(void)
{

    int row;
    int column;

    // Read csv file into 2 dimensional array
    int **data = readcsv("/home/jelly/Downloads/mnist_train.csv", &row, &column);

    // Transpose data array
    double **dataT = transpose(&row, &column, data);

    // free unused matrix
    free_mat(data, column);

    // assign data into training data & freeing unused data
    double **data_train = assign_values(dataT, 0, row, 0, column);
    free_mat(dataT, row);

    // Seperate pixel data and labels
    double **Y_train = assign_values(data_train, 0, 1, 0, column);
    double **X_train = assign_values(data_train, 1, row, 0, column);

    // Normalize pixel data
    for (int i = 0; i < row - 1; i++)
    {
        for (int j = 0; j < column - 0; j++)
        {
            X_train[i][j] = X_train[i][j] / 255;
        }
    }

    // initialize layers

    double **W1 = Xavier_init(W1x, 784);
    double **b1 = zeros(W1x, 1);
    double **W2 = Xavier_init(W2x, W1x);
    double **b2 = zeros(W2x, 1);

    // Forward Prop
    double **dotz1 = dot(W1x, row - 1, row - 1, column, W1, X_train);
    double **Z1 = boradcast(W1x, column, W1x, 1, dotz1, b1);
    free_mat_double(dotz1, W1x);
    double **A1 = Leaky_relu(Z1, W1x, column);
    double **dotz2 = dot(W2x, W1x, W1x, column, W2, A1);
    double **Z2 = boradcast(W2x, column, W2x, 1, dotz2, b2);
    free_mat_double(dotz2, W2x);
    double **A2 = softmax(Z2, W2x, column);

    // Backward Prop
    double **ohy = One_Hot(Y_train, 1, column);
    double **dZ2 = matrix_arithmetic(A2, ohy, 1, column, 'm');
    double** A1T = transpose_wo_change(W1x, column, A1);
    double** dW2 = dot(1, column, column, W1x, dZ2, A1T);
    double db2 = 0;
    for (int i = 0; i <column; i++)
    {
        db2 += dZ2[0][i];
    }
    db2 = db2/60000;
    for (int i = 0; i < W1x; i++)
    {
        dW2[0][i] *= 1/60000;
    }
    double** redevZ1 = leaky_relu_deriv(Z1, W1x, column);
    double** dZ1 = matrix_arithmetic(redevZ1, 




    free_mat(A1T, column);



    printMatrix(b1, 10, 1);
    free_mat_double(W1, W1x);
    free_mat_double(b1, W1x);
    free_mat_double(W2, W2x);
    free_mat_double(b2, W2x);
    free_mat_double(Z1, W1x);
    free_mat_double(A1, W1x);
    free_mat_double(Z2, W2x);
    free_mat_double(A2, W2x);
    free_mat_double(ohy, 1);
    free_mat_double(dZ2, 1);
    free_mat_double(data_train, row);

    free_mat_double(Y_train, 1);

    free_mat_double(X_train, row - 1);

    return 0;
}
