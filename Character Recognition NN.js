 
// Port of Character recognition neural network from here:
// https://github.com/CodingTrain/Toy-Neural-Network-JS/tree/master/examples/mnist
// with many modifications 


// defined for the MNIST
const PIXELS        = 28;                       // images in data set are tiny 
const PIXELSSQUARED = PIXELS * PIXELS;

// number of training and test exemplars in the data set:
const NOTRAIN = 60000;
const NOTEST  = 10000;

// no of nodes in network 
const noinput  = PIXELSSQUARED;
const nohidden = 64;
const nooutput = 10;

// code added to crop the doodle image canvas pixel area
const PIXELS_DROP = 24;

const learningrate = 0.1;   // default 0.1  

// should we train every timestep or not 
let do_training = true;

// how many to train and test per timestep 
const TRAINPERSTEP = 30;
const TESTPERSTEP  = 5;

// multiply it by this to magnify for display 
const ZOOMFACTOR    = 9;                        
const ZOOMPIXELS    = ZOOMFACTOR * PIXELS; 

// 3 rows of
// large image + 50 gap + small image    
// 50 gap between rows 

const canvaswidth = ( PIXELS + ZOOMPIXELS ) + 50;
const canvasheight = ( ZOOMPIXELS * 3 ) + 100;


const DOODLE_THICK = 16;    // thickness of doodle lines 
const DOODLE_BLUR = 1;      // Removing the blur factor to improve the CNN model accuracy

let mnist;      
// all data is loaded into this 
// mnist.train_images
// mnist.train_labels
// mnist.test_images
// mnist.test_labels


let nn;

let trainrun = 1;
let train_index = 0;

let testrun = 1;
let test_index = 0;
let total_tests = 0;
let total_correct = 0;

// images in LHS:
let doodle, demo;
let doodle_exists = false;
let demo_exists = false;

let mousedrag = false;      // are we in the middle of a mouse drag drawing?  


// save inputs to global var to inspect
// type these names in console 
var train_inputs, test_inputs, demo_inputs, doodle_inputs;


// Matrix.randomize() is changed to point to this. Must be defined by user of Matrix. 

function randomWeight()
{
   return ( AB.randomFloatAtoB ( -0.5, 0.5 ) );
   // Coding Train default is -1 to 1
}    


// CSS trick 
// make run header bigger 
 $("#runheaderbox").css ( { "max-height": "95vh" } );



//--- start of AB.msgs structure: ---------------------------------------------------------
// We output a serious of AB.msgs to put data at various places in the run header 
var thehtml;

  // 1 Doodle header 
  thehtml = "<hr> <h1> 1. Doodle </h1> Top row: Doodle (left) and shrunk (right). <br> " +
        " Draw your doodle in top LHS. <button onclick='wipeDoodle();' class='normbutton' >Clear doodle</button> <br> ";
   AB.msg ( thehtml, 1 );

  // 2 Doodle variable data (guess)
  
  // 3 Training header
  thehtml = "<hr> <h1> 2. Training </h1> Middle row: Training image magnified (left) and original (right). <br>  " +
        " <button onclick='do_training = false;' class='normbutton' >Stop training</button> <br> ";
  AB.msg ( thehtml, 3 );
     
  // 4 variable training data 
  
  // 5 Testing header
  thehtml = "<h3> Hidden tests </h3> " ;
  AB.msg ( thehtml, 5 );
           
  // 6 variable testing data 
  
  // 7 Demo header 
  thehtml = "<hr> <h1> 3. Demo </h1> Bottom row: Test image magnified (left) and  original (right). <br>" +
        " The network is <i>not</i> trained on any of these images. <br> " +
        " <button onclick='makeDemo();' class='normbutton' >Demo test image</button> <br> ";
   AB.msg ( thehtml, 7 );
   
  // 8 Demo variable data (random demo ID)
  // 9 Demo variable data (changing guess)
  
const greenspan = "<span style='font-weight:bold; font-size:x-large; color:darkgreen'> "  ;

//--- end of AB.msgs structure: ---------------------------------------------------------


const SOUND_ALARM  = '/uploads/jai/audio1.wav' ;
var alarm = new Audio ( SOUND_ALARM );

function setup() 
{
  
  createCanvas ( canvaswidth, canvasheight );
  doodle = createGraphics ( ZOOMPIXELS, ZOOMPIXELS );       // doodle on larger canvas 
  doodle.pixelDensity(1);
  
  // JS load other JS 
  // maybe have a loading screen while loading the JS and the data set 
  
  AB.loadingScreen();
  alarm.play();							// play once, no loop

 $.getScript ( "/uploads/codingtrain/matrix.js", function(){
    $.getScript ( "/uploads/codingtrain/mnist.js", function(){
        $.getScript ( "/uploads/jai/nn.js", function(){                     // updating the nn.js file to have the other activation functions tried and tested
            $.getScript ( "/uploads/jai/utils.js", function(){              // loading the additional utility function for implementing the CNN network
                $.getScript ( "/uploads/jai/cnn.js", function(){            // loading the CNN network implemented
                    $.ajax({
                        url: "/uploads/jai/cnn_accuracy.json",              // loading the pre-defined weights for the CNN network
                        dateType: "json",
                        success: JSONLoaded
                    });
                });
                
            });
        });
   });
 });
}



// load data set from local file (on this server)
function loadData()    
{
  loadMNIST ( function(data)    
  {
    mnist = data;
    console.log ("All data loaded into mnist object:")
    console.log(mnist);
    AB.removeLoading();     // if no loading screen exists, this does nothing 
  });
}



function getImage ( img )      // make a P5 image object from a raw data array   
{
    let theimage  = createImage (PIXELS, PIXELS);    // make blank image, then populate it 
    theimage.loadPixels();        
    
    for (let i = 0; i < PIXELSSQUARED ; i++) 
    {
        let bright = img[i];
        let index = i * 4;
        theimage.pixels[index + 0] = bright;
        theimage.pixels[index + 1] = bright;
        theimage.pixels[index + 2] = bright;
        theimage.pixels[index + 3] = 255;
    }
    
    theimage.updatePixels();
    return theimage;
}


function getInputs ( img )      //convert imgage array into normalised input array 
{
    let inputs = [];
    for (let i = 0; i < PIXELSSQUARED ; i++){
        let bright = img[i];
        inputs[i] = bright / 255;       // normalise to 0 to 1
    } 
    return ( inputs );
}

 

function trainit (show)        // train the network with a single exemplar, from global var "train_index", show visual on or off 
{
  let img   = mnist.train_images[train_index];
  let label = mnist.train_labels[train_index];
  
  // optional - show visual of the image 
  if (show)                
  {
    var theimage = getImage ( img );    // get image from data array 
    image ( theimage,   0,                ZOOMPIXELS+50,    ZOOMPIXELS,     ZOOMPIXELS  );      // magnified 
    image ( theimage,   ZOOMPIXELS+50,    ZOOMPIXELS+50,    PIXELS,         PIXELS      );      // original
  }

  // set up the inputs
  let inputs = getInputs ( img );       // get inputs from data array 

  // set up the outputs
  let targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  targets[label] = 1;       // change one output location to 1, the rest stay at 0 

  train_inputs = inputs;        // can inspect in console 
  nn.train ( inputs, targets );

  thehtml = " trainrun: " + trainrun + "<br> no: " + train_index ;
  AB.msg ( thehtml, 4 );

  train_index++;
  if ( train_index == NOTRAIN ) 
  {
    train_index = 0;
    console.log( "finished trainrun: " + trainrun );
    trainrun++;
  }
}


function testit()    // test the network with a single exemplar, from global var "test_index"
{ 
  let img   = mnist.test_images[test_index];
  let label = mnist.test_labels[test_index];

  // set up the inputs
  let inputs = getInputs ( img ); 
  
  test_inputs = inputs;        // can inspect in console 
  let prediction    = nn.predict(inputs);       // array of outputs 
  let guess         = findMax(prediction);      // the top output 

  total_tests++;
  if (guess == label)  total_correct++;

  let percent = (total_correct / total_tests) * 100 ;
  
  thehtml =  " testrun: " + testrun + "<br> no: " + total_tests + " <br> " +
        " correct: " + total_correct + "<br>" +
        "  score: " + greenspan + percent.toFixed(2) + "</span>";
  AB.msg ( thehtml, 6 );

  test_index++;
  if ( test_index == NOTEST ) 
  {
    console.log( "finished testrun: " + testrun + " score: " + percent.toFixed(2) );
    testrun++;
    test_index = 0;
    total_tests = 0;
    total_correct = 0;
  }
}




//--- find no.1 (and maybe no.2) output nodes ---------------------------------------
// (restriction) assumes array values start at 0 (which is true for output nodes) 

function find12(pValue) // return array showing indexes of no.1 and no.2 values in array 
{
  let a = 0;
  let b = 0;
  let aValue = 0;
  let bValue = 0;
  for (let i = 0; i < 10; i++) {
	let predictedVal = pValue[0].getValue(0, 0, i);
    if (predictedVal > aValue) {
      a = i;
	  aValue = predictedVal;
    }
  }
  for (let i = 0; i < 10; i++) {
	let predictedVal = pValue[0].getValue(0, 0, i);
	if ((a != i) && (predictedVal > bValue)) {
      b = i;
      bValue = predictedVal;
    }
  }
  return [a, b];
}


// just get the maximum - separate function for speed - done many times 
// find our guess - the max of the output nodes array

function findMax (a)        
{
  let no1 = 0;
  let no1value = 0;     
  
  for (let i = 0; i < a.length; i++) 
  {
    if (a[i] > no1value) 
    {
      no1 = i;
      no1value = a[i];
    }
  }
  
  return no1;
}


var gif_loadImg, gif_createImg;

function preload() {
  gif_loadImg = loadImage("/uploads/jai/EllipticalCostlyChrysomelid-size_restricted.gif");
  gif_createImg = createImg("/uploads/jai/EllipticalCostlyChrysomelid-size_restricted.gif");
}



// --- the draw function -------------------------------------------------------------
// every step:
 
function draw() 
{

  // loads only first frame
  image(gif_loadImg, 900, 80);
  
  // updates animation frames by using an html
  // img element, positioning it over top of
  // the canvas.
  gif_createImg.position(900, 80);
  
  
  // check if libraries and data loaded yet:
  if ( typeof mnist == 'undefined' ) return;


// how can we get white doodle on black background on yellow canvas?
//        background('#ffffcc');    doodle.background('black');

      background ('black');
    
if ( do_training )    
{
  // do some training per step 
    for (let i = 0; i < TRAINPERSTEP; i++) 
    {
      if (i == 0)    trainit(true);    // show only one per step - still flashes by  
      else           trainit(false);
    }
    
  // do some testing per step 
    for (let i = 0; i < TESTPERSTEP; i++) 
      testit();
}

  // keep drawing demo and doodle images 
  // and keep guessing - we will update our guess as time goes on 
  
  if ( demo_exists )
  {
    drawDemo();
    guessDemo();
  }
  if ( doodle_exists ) 
  {
    drawDoodle();
    guessDoodle();
  }


// detect doodle drawing 
// (restriction) the following assumes doodle starts at 0,0 

  if ( mouseIsPressed )         // gets called when we click buttons, as well as if in doodle corner  
  {
     // console.log ( mouseX + " " + mouseY + " " + pmouseX + " " + pmouseY );
     var MAX = ZOOMPIXELS + 20;     // can draw up to this pixels in corner 
     if ( (mouseX < MAX) && (mouseY < MAX) && (pmouseX < MAX) && (pmouseY < MAX) )
     {
        mousedrag = true;       // start a mouse drag 
        doodle_exists = true;
        doodle.stroke('white');
        doodle.strokeWeight( DOODLE_THICK );
        doodle.line(mouseX, mouseY, pmouseX, pmouseY);      
     }
  }
  else 
  {
      // are we exiting a drawing
      if ( mousedrag )
      {
            mousedrag = false;
            // console.log ("Exiting draw. Now blurring.");
            doodle.filter (BLUR, DOODLE_BLUR);    // just blur once 
            //   console.log (doodle);
      }
  }
}

// Code implemented considering the Adam Smith webCNN GitHub repository
// CNN Neural Network used to guess Doodle hand-written image recognition
function JSONLoaded(response){
    
    // new model implementation using CNN
    cnnFromJSON(response);
    console.log("JSON Loaded!");
    console.log(response);
    
    // old model implementation for training the data and testing the demo image 
    console.log ("All JS loaded");
    nn = new NeuralNetwork(  noinput, nohidden, nooutput );
    nn.setLearningRate ( learningrate );
    loadData();
}

// Loading the CNN network from the "cnn_accuracy" JSON file
function cnnFromJSON( networkJSON )
{
	cnn = new WebCNN();

	if ( networkJSON.momentum != undefined ){
		cnn.setMomentum( networkJSON.momentum );
	}
	if ( networkJSON.lambda != undefined ){
		cnn.setLambda( networkJSON.lambda );
	}
	if ( networkJSON.learningRate != undefined ){
		cnn.setLearningRate( networkJSON.learningRate );
	}

	for ( var layerIndex = 0; layerIndex < networkJSON.layers.length; ++layerIndex ){
		let layerDesc = networkJSON.layers[ layerIndex ];
		console.log( layerDesc );
		cnn.newLayer( layerDesc );
	}

	for ( var layerIndex = 0; layerIndex < networkJSON.layers.length; ++layerIndex ){
		let layerDesc = networkJSON.layers[ layerIndex ];

		switch ( networkJSON.layers[ layerIndex ].type ){
			case LAYER_TYPE_CONV:
			case LAYER_TYPE_FULLY_CONNECTED:
			{
				if ( layerDesc.weights != undefined && layerDesc.biases != undefined )
				{
					cnn.layers[ layerIndex ].setWeightsAndBiases( layerDesc.weights, layerDesc.biases );
				}
				break;
			}
		}
	}

	cnn.initialize();
}

// to bring the data in required format to feed the CNN model
function requireFormat(image,size){
    return {
        "width": size,
        "height": size,
        "data": getImage(randomCrop(image,size),size).pixels
    };
}



function centerImage(pixels, size){
    // convert to matrix
    let m = [];
    for(let i = 0; i < size; i++){
        m[i] = [];
        for(let j = 0; j < size; j++){
            m[i][j] = pixels[(i*size + j) * 4];
        }
    }
    
    // To compute the bounding box of the doodle image canvas
    var topmost = Number.MAX_VALUE;
    var leftmost = Number.MAX_VALUE;
    var bottommost = -1;
    var rightmost = -1;
    for(var y = 0; y < m.length; y++){
        var l = m[y].indexOf(255);
        var r = m[y].lastIndexOf(255);
        if (l >= 0 && l < leftmost) leftmost = l;       // setting the leftmost position
        if (r >= 0 && r > rightmost) rightmost = r;     // setting the rightmost position
        if (l >= 0 && y < topmost) topmost = y;         // setting the topmost position
        if (l >= 0 && y > bottommost) bottommost = y;   // setting the bottommost position
    }
    
    // translate to centre
    let transY = Math.floor((size - bottommost - topmost) / 2);
    let transX = Math.floor((size - rightmost - leftmost) / 2);
    let result = Array(size).fill().map(() => Array(size).fill(0));
    for(i = topmost; i <= bottommost; i++)
        for(j = leftmost; j <= rightmost; j++)
            result[i + transY][j + transX] = m[i][j];
        
    //convert back to 1D array
    let m1D = [];
    for(let i = 0; i < size; i++)
        for (let j = 0; j < size; j++)
            m1D[i*size + j] = result[i][j];
        
    return m1D;
}


// convert image array into normalized input array
function randomCrop(img, size){
    const maxStartIndex = PIXELS - size;
    let xRand = Math.floor( Math.random() * maxStartIndex);
    let yRand = Math.floor( Math.random() * maxStartIndex);
    return crop(img, size, xRand, yRand);
}


// Crop of size * size part from image starting at (X, Y)
function crop(img, size, x = 2, y = 2){
    const PIXELS_DROP = size;
    let xEnd = x + PIXELS_DROP;
    let yEnd = y + PIXELS_DROP;
    let inputs = [];
    for (let i = x; i < xEnd; i++)
        for (let j = y; j < yEnd; j++)
            inputs.push(img[ i * PIXELS + j ]);
    return(inputs);
}


//--- demo -------------------------------------------------------------
// demo some test image and predict it
// get it from test set so have not used it in training


function makeDemo()
{
    demo_exists = true;
    var  i = AB.randomIntAtoB ( 0, NOTEST - 1 );  
    
    demo        = mnist.test_images[i];     
    var label   = mnist.test_labels[i];
    
   thehtml =  "Test image no: " + i + "<br>" + 
            "Classification: " + label + "<br>" ;
   AB.msg ( thehtml, 8 );
   
   // type "demo" in console to see raw data 
}


function drawDemo()
{
    var theimage = getImage ( demo );
     //  console.log (theimage);
     
    image ( theimage,   0,                canvasheight - ZOOMPIXELS,    ZOOMPIXELS,     ZOOMPIXELS  );      // magnified 
    image ( theimage,   ZOOMPIXELS+50,    canvasheight - ZOOMPIXELS,    PIXELS,         PIXELS      );      // original
}


function guessDemo()
{
   let inputs = getInputs ( demo ); 
   
  demo_inputs = inputs;  // can inspect in console 
  
  let prediction    = nn.predict(inputs);       // array of outputs 
  let guess         = findMax(prediction);      // the top output 

   thehtml =   " We classify it as: " + greenspan + guess + "</span>" ;
   AB.msg ( thehtml, 9 );
}




//--- doodle -------------------------------------------------------------

function drawDoodle()
{
    // doodle is createGraphics not createImage
    let theimage = doodle.get();
    // console.log (theimage);
    
    image ( theimage,   0,                0,    ZOOMPIXELS,     ZOOMPIXELS  );      // original 
    image ( theimage,   ZOOMPIXELS+50,    0,    PIXELS,         PIXELS      );      // shrunk
}
      
      
function guessDoodle() 
{
   // doodle is createGraphics not createImage
   let img = doodle.get();
  
  img.resize ( PIXELS, PIXELS );     
  img.loadPixels();

  // set up inputs   
  let inputs = [];
  for (let i = 0; i < PIXELSSQUARED ; i++) 
  {
     inputs[i] = img.pixels[i * 4] / 255;
  }
  
  doodle_inputs = inputs;     // can inspect in console 

  // feed forward to make prediction 
  //let prediction    = nn.predict(inputs);       // array of outputs 
  
  // Guessing the doodle using the CNN approach
  let prediction = cnn.classifyImages([requireFormat(centerImage(img.pixels,PIXELS), PIXELS_DROP)]);
  let b             = find12(prediction);       // get no.1 and no.2 guesses  

  thehtml =   " We classify it as: " + greenspan + b[0] + "</span> <br>" +
            " No.2 guess is: " + greenspan + b[1] + "</span>";
  AB.msg ( thehtml, 2 );
}


function wipeDoodle()    
{
    doodle_exists = false;
    doodle.background('black');
}


// --- debugging --------------------------------------------------
// in console
// showInputs(demo_inputs);
// showInputs(doodle_inputs);


function showInputs ( inputs )
// display inputs row by row, corresponding to square of pixels 
{
    var str = "";
    for (let i = 0; i < inputs.length; i++) 
    {
      if ( i % PIXELS == 0 )    str = str + "\n";                                   // new line for each row of pixels 
      var value = inputs[i];
      str = str + " " + value.toFixed(2) ; 
    }
    console.log (str);
}
