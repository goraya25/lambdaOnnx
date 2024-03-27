/**
 *
 * Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format
 * @param {Object} event - API Gateway Lambda Proxy Input Format
 *
 * Context doc: https://docs.aws.amazon.com/lambda/latest/dg/nodejs-prog-model-context.html 
 * @param {Object} context
 *
 * Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
 * @returns {Object} object - API Gateway Lambda Proxy Output Format
 * 
 */
import * as ort from "onnxruntime-node";
import * as Jimp from 'jimp'
//import {ImageLoader} from '@loaders.gl/images';
//import ndarray from 'ndarray'
import _ from 'lodash';
//import {createCanvas, loadImage} from 'canvas';
//import onnx from 'onnxjs'
//import JSDOM from 'jsdom'

export const lambdaHandler = async (event, context) => {
  const modelPath = "./best2.onnx";
  //const { document } = (new JSDOM(`...`)).window;
  //var image = loadImage('./download.jpg')
  //image.src = "./download.jpg";
  const session = await ort.InferenceSession.create('./best2.onnx');
  const dims = [1, 3, 640, 640];
  var image = await Jimp.default.read('./download.jpg').then((imageBuffer) => {
    return imageBuffer.resize(640, 640);
  });
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
   redArray.push(imageBufferData[i]);
   greenArray.push(imageBufferData[i + 1]);
   blueArray.push(imageBufferData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i, l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
   float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const preprocessedData = new ort.Tensor("float32", float32Data, dims);
  //const preprocessedData = new ort.Tensor("float32", image, dims);

  const feeds = {};
  feeds[session.inputNames[0]] = preprocessedData;
  // Run the session inference.
  const outputData = await session.run(feeds);

  const output = outputData[session.outputNames[0]];
  const slicedOutput = Array.prototype.slice.call(output.data)
  //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  //var outputSoftmax = softmax(Array.prototype.slice.call(output.data));
  //Get the top 5 results.
  
  const sortedArr = slicedOutput.sort();
  const slicedArray = sortedArr.slice(sortedArr.length-5, sortedArr.length);

  
  const response = {
    statusCode: 200,
    body: JSON.stringify({
      message: slicedArray,
    })
  };

  return response;
};