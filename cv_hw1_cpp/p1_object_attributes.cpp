#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cstring>

// Binarize the gray_image.

class Union_Set
{
    public:
        std::vector<uint16_t> parent;
        uint16_t size;

        Union_Set()
        {
            size = 10;
            parent = std::vector<uint16_t>(size,0);
            for(uint16_t i=0;i<parent.size();i++)
                parent[i]=i;
        }
        Union_Set(uint16_t label)
        {
            size = label;
            parent = std::vector<uint16_t>(size);
            for(uint16_t i=0;i<parent.size();i++)
                parent[i]=i;
        }
        uint16_t find(uint16_t x)
        {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        }
        void Union(uint16_t cur, uint16_t label)
        {
            if(cur >= size || label >= size)
            {
                uint16_t original_size = size;
                size = (cur >= label) ? cur * 2 : label * 2;
                parent.resize(size);
                for(uint16_t i=original_size;i<size;i++)
                    parent[i]=i;
            }
            
            parent[find(label)] = find(cur);
        }
};



cv::Mat binarize(cv::Mat gray_image, int threshval)
{
    try
    {
        assert(gray_image.dims == 2);
    }
    catch(const std::exception& e)
    {
        std::cerr << "The dim of the gray_image is not 2! " << '\n';
        exit(-1);
    }
    int rows=gray_image.rows;
    int cols=gray_image.cols;
    // Copy the data to binary_image.
    cv::Mat binary_image;
    gray_image.copyTo(binary_image);
    for(int i=0; i<rows; i++)
    {
        uint8_t *rowi=binary_image.ptr<uint8_t>(i);
        for(int j=0; j<cols; j++)
        {
            if (rowi[j] <= threshval)
                rowi[j] = 0;
            else
                rowi[j] = 255;
        }
    }
    return binary_image;
}

cv::Mat label(cv::Mat binary_image)
{
    // Finding neighbors.
    std::function<std::vector<uint16_t>(uint16_t,uint16_t,cv::Mat)> prior_neighbors = [](uint16_t i,uint16_t j, cv::Mat labeled_image)
    {
        std::vector<uint16_t> neighbors;
        if(i > 0)
        {
            // read the element of the labeled image.
            const uint16_t *const rowi_minus = labeled_image.ptr<uint16_t>(i-1);
            if(rowi_minus[j] > 0)
                neighbors.push_back(rowi_minus[j]);
        }
        if(j > 0)
        {
            const uint16_t *const rowi = labeled_image.ptr<uint16_t> (i);
            if(rowi[j-1] > 0)
                neighbors.push_back(rowi[j-1]);
        }
        return neighbors;
    };
    // Initialize the union set.
    Union_Set parents;
    int rows = binary_image.rows, cols = binary_image.cols;
    cv::Mat labeled_image(rows,cols,CV_16U,cv::Scalar(0));
    uint16_t label_counter = 0;

    // The first pass.
    for(int i=0;i<rows;i++)
    {
        const uint16_t* binary_image_rowi = binary_image.ptr<uint16_t>(i);
        uint16_t* labeled_image_rowi = labeled_image.ptr<uint16_t>(i);
        for(int j=0;j<cols;j++)
        {
            if(binary_image_rowi[j] == 0)
                continue;
            std::vector<uint16_t> neighbors = prior_neighbors(i,j,labeled_image);
            if(neighbors.size()==0)
            {
                label_counter += 1;
                // write to the labeled image.
                labeled_image.at<uint16_t>(i,j)=label_counter;
            }
            else
            {
                labeled_image_rowi[j] = *(std::min_element(neighbors.begin(),neighbors.end()));
                uint16_t cur = labeled_image_rowi[j];
                for(uint16_t label: neighbors)
                {
                    if(label != labeled_image_rowi[j])
                        parents.Union(cur, label);
                }
            }
        }
    }
    // The second pass.
    uint16_t maximum_label_level=0;
    for(int i=0;i<rows;i++)
    {
        const uint16_t* binary_image_rowi = binary_image.ptr<uint16_t>(i);
        uint16_t* labeled_image_rowi = labeled_image.ptr<uint16_t>(i);
        for(int j=0;j<cols;j++)
        {
            uint16_t currentLabel = (uint16_t)parents.find(labeled_image_rowi[j]);
            labeled_image.at<uint16_t>(i,j) = currentLabel;
            maximum_label_level = (maximum_label_level > currentLabel) ? maximum_label_level : currentLabel;
        }
    }
    return labeled_image;
}

void get_attribute(cv::Mat labeled_image)
{
    int height=labeled_image.rows,width = labeled_image.cols;
    std::set <uint16_t> label_level; // Hash set for adding numbers.
    for (int i = 0; i < height; i++)
    {
        // labeled image should be read only.
        const uint16_t *labeled_image_rowi = labeled_image.ptr<uint16_t>(i);
        for (int j = 0; j < width; j++)
        {
            uint16_t current_label = labeled_image_rowi[j];
            if (label_level.find(current_label) == label_level.end() && current_label != 0)
                label_level.insert(current_label);  // Find non-repeat label to insert.
        }
    }
    std::vector<cv::Mat> picture_list;
    for(int label: label_level)
    {
        cv::Mat current_label_img(height,width,CV_16U,cv::Scalar(0));   // Initialize the image.
        for(int i = 0; i < height; i++)
        {
            const uint16_t *labeled_image_rowi = labeled_image.ptr<uint16_t>(i);
            for(int j = 0;j<width; j++)
            {
                uint16_t current_label = labeled_image_rowi[j];
                if (current_label != label)
                    current_label_img.at<uint16_t> (i,j) = 0;
                else
                    current_label_img.at<uint16_t> (i,j) = label;
            }
        }
        picture_list.push_back(current_label_img);
    }
    
}


int main(int argc, char *argv[])
{
    // The argument detect.
    if (argc < 2)
    {
        std::cerr << "Usage: ./program1 <thresh_val>\n" << std::endl;
        exit(-1); 
    }

    // Get the thershold value.
    int thresh_val = atoi(argv[1]);

    // Confirm the input directory and the pictures.
    std::string input_dir = "./data/";
    std::string pic1 = "many_objects_1.png";
    std::string pic2 = "many_objects_2.png";
    std::string pic3 = "two_objects.png";

    // Convert into the cv2 matrix.
    std::vector<std::string> pictures = {pic1,pic2,pic3};
    for(auto &pic:pictures)
    {
        cv::Mat img = cv::imread(input_dir + pic,cv::IMREAD_COLOR);
        cv::Mat gray_image;
        cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
        cv::Mat binary_image = binarize(gray_image,thresh_val);
        // uint16 pictures for label.
        binary_image.convertTo(binary_image,CV_16U);
        cv::Mat labeled_image = label(binary_image);
        cv::imwrite("./output/" + pic + "_gray.png", gray_image);
        cv::imwrite("./output/" + pic + "_binary.png", binary_image);
        cv::Mat saving_labeled_image;
        labeled_image.convertTo(saving_labeled_image,CV_8UC1);
        cv::imwrite("./output/" + pic + "_labeled.png", saving_labeled_image);
        get_attribute(labeled_image);
    }
}