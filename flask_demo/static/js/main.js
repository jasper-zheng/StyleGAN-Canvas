$(document).ready(function () {
  const FRAME_SIZE    = 352   // input frame size (left)
  const FRAME_SIZE_Y  = 640   // input frame size (left)
  let crop_factor     = 0.3    // 0: no crop, 1: crop everything
  const input_quality = 0.5  // quality from client to server
  const FRAME_RATE    = 120   // ms per frame

  let namespace = "/demo";
  let video = document.querySelector("#videoElement");
  let canvas = document.querySelector("#inputCanvas");
  canvas.width = FRAME_SIZE;
  canvas.height = FRAME_SIZE_Y;
  // canvas.style.right = FRAME_SIZE + 10 + "px";
  canvas.style.right = 10 + "px";
  
  // canvas.style.filter = "brightness(1.1)"
  // canvas.style.filter = "contrast(10%);"
  let ctx = canvas.getContext('2d');
  ctx.translate(FRAME_SIZE,0);
  ctx.scale(-1,1);
  // ctx.filter = "saturate(10%) brightness(0.2) contrast(10%);"
  var constraints = {
    audio: false,
    video: {
      width: FRAME_SIZE,
      height: FRAME_SIZE_Y,
    }
  };
    
  let canvasContainer = document.querySelector("#canvasContainer");
  canvasContainer.style.width = 99 + FRAME_SIZE + "px";
  // canvasContainer.style.width = 99 + FRAME_SIZE*2 + "px";
  canvasContainer.style.height = 45 + FRAME_SIZE_Y + "px";
  canvasContainer.style.left = 30 + FRAME_SIZE*2 + "px";
  let displays = document.querySelector("#displays");
  displays.style.height = FRAME_SIZE_Y*2 + "px";
  displays.style.width = FRAME_SIZE*3+ 109 + "px";
    
  // ctx.filter = 'blur(6px)';
  ctx.filter = 'saturate(120%) brightness(1) contrast(100%)';
  let config_update = "";
  var configs_template = {
    "angle": 0,
    "translateX": 0,
    "translateY": 0,
    "scale": 1,
    "erosion": 0,
    "dilation": 0,
    "multiply": 1,
    "cluster": [0,1,2,3,4]
  }
  var configs = {}

  let layer_names = document.querySelector("#layerNames");
  let initialisation = false;
  let layer_selection = '';
  let layer_list = {};
  let cluster_numbers = {};

  let cur_input = 0  // 0: webcam, 1:file
  let file_is_init = false
  let is_pause = false

  output_canvas = document.getElementById('outputCanvas');
  output_canvas.style.width = FRAME_SIZE*2 + "px";
  output_canvas.style.height = FRAME_SIZE_Y*2 + "px";
  // output_canvas.style.left = 109+FRAME_SIZE + "px";
  output_canvas.style.left = "0px";
    
    
    // output_canvas.style.left = 109+FRAME_SIZE*2 + "px";
  // output_canvas_x = document.getElementById('inputCanvasX');
  // output_canvas_x.style.width = FRAME_SIZE + "px";
  // output_canvas_x.style.height = FRAME_SIZE_Y + "px";
  var localMediaStream = null;

  var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);

  let pauseBtn = document.querySelector("#pauseBtn")
  pauseBtn.style.right = (FRAME_SIZE - 48)/2 + "px";
  pauseBtn.onclick = function() {
    if (is_pause) {
      is_pause = false
      pauseBtn.innerHTML = 'Pause'
    } else {
      is_pause = true
      pauseBtn.innerHTML = 'Resume'
    }
  }
  let webcamBtn = document.querySelector("#webcamBtn")
  webcamBtn.style.height = (44 + FRAME_SIZE_Y)/2 + "px";
  let fileBtn = document.querySelector("#fileBtn")
  fileBtn.style.height = (44 + FRAME_SIZE_Y)/2 + "px";
  let uploadBtn = document.querySelector("#fileUploadContainer")
  uploadBtn.style.right = (FRAME_SIZE - 88)/2 + "px";
  webcamBtn.onclick = function() {
    if (cur_input==1) {
      cur_input = 0
      webcamBtn.setAttribute('class', 'inputBtn inputBtnAct')
      fileBtn.setAttribute('class', 'inputBtn inputBtnDisact')
      pauseBtn.style.display = 'block'
      uploadBtn.style.display = 'none'
      ctx.translate(FRAME_SIZE_Y,0);
      ctx.scale(-1,1);
    }
  }
  fileBtn.onclick = function() {
    if (cur_input==0) {
      cur_input = 1
      webcamBtn.setAttribute('class', 'inputBtn inputBtnDisact')
      fileBtn.setAttribute('class', 'inputBtn inputBtnAct')
      pauseBtn.style.display = 'none'
      uploadBtn.style.display = 'block'
      ctx.translate(FRAME_SIZE_Y,0);
      ctx.scale(-1,1);
    }
  }
  let loadedImg = new Image();
  let uploadBtnInput = document.querySelector("#fileUpload")
  fileUpload.onchange = function(e) {
    if (!file_is_init){
      file_is_init = true
    }
    console.log(URL.createObjectURL(this.files[0]))
    loadNewImgToCanvas(URL.createObjectURL(this.files[0]))
  }
  function loadImage(url) {
    return new Promise(r => { let i = new Image(); i.onload = (() => r(i)); i.src = url; });
  }
  async function loadNewImgToCanvas(p){
    loadedImg = await loadImage(p);
    // ctx.drawImage(img, 0,0,img.imageWidth, img.imageHeight, 0,0,FRAME_SIZE_Y,FRAME_SIZE);
  }

  function sendFrame() {

    if (!localMediaStream) {
      return;
    }

    if (config_update){
      if (layer_list[layer_selection][0]){
        console.log('update ' + layer_selection)
        socket.emit('config_update', layer_selection, configs[layer_selection]);
      }
      config_update = "";
    }

    if (cur_input==0){
      if (!is_pause){
        ctx.drawImage(video,
                      crop_factor/2*video.videoWidth,
                      crop_factor/2*video.videoHeight,
                      (1-crop_factor/2)*video.videoWidth,
                      (1-crop_factor/2)*video.videoHeight,
                      0,0,FRAME_SIZE,FRAME_SIZE_Y);
      }
    } else if (cur_input==1) {
      if (file_is_init){
        ctx.drawImage(loadedImg,
                      0,0,loadedImg.width,loadedImg.height,FRAME_SIZE,0,FRAME_SIZE,FRAME_SIZE_Y);
      } else {

      }
    }

    let dataURL = canvas.toDataURL('image/jpeg',input_quality);
    socket.emit('input_frame', dataURL);

  }

  socket.on('connect', function() {
    console.log('Connected!');
  });

  socket.on('set_layer_names',function(data){
    // output_canvas.setAttribute('src', data.image_data);
    console.log(data.names)
    if (!initialisation){
      for (let i=0;i<data.names.length;i++){
        // var name = data.names[i]
        configs[data.names[i]] = {...configs_template}
        configs[data.names[i]]['translate'] = [configs[data.names[i]]["translateX"], configs[data.names[i]]["translateY"]];

        var div = document.createElement('div');
        var name_div = document.createElement('div');
        var selectionBtn = document.createElement('button');
        var label_div = document.createElement('div');
        name_div.innerHTML = data.names[i]
        var labels = data.names[i].split('_')
        var idx = labels[0][1]
        label_div.innerHTML = 'idx: '+labels[0][1]+'<br>res: '+(parseInt(labels[1])-20)+'<br>channels: '+labels[2]
        div.setAttribute('class', 'layerNamesDiv layerDisSelected')
        name_div.setAttribute('class', 'layerNamesText')
        selectionBtn.setAttribute('class', 'layerSelectionBtn layerSelectionBtnDown')
        label_div.setAttribute('class', 'layerNamesLabel')

        selectionBtn.addEventListener("click", function(){
          switchLayerActivation(data.names[i])
        })
        div.addEventListener("click", function() {
          updateSelection(data.names[i]);
        });
        div.appendChild(selectionBtn)
        div.appendChild(name_div)
        div.appendChild(label_div)
        layer_names.appendChild(div)
        layer_list[data.names[i]] = [false, div]
        cluster_numbers[data.names[i]] = 5
      }
      layer_list[data.names[1]][0] = false

      layer_selection = data.names[0]
      initialisation = true;
      console.log(cluster_numbers)
      updateSelection(data.names[1])
    }

  });


  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    localMediaStream = stream;

    setInterval(function () {
      sendFrame();
    }, FRAME_RATE);

    socket.on('processed_frame',function(data){
        output_canvas.style.backgroundImage = `url(${data.image_data})`;
        // output_canvas_x.style.backgroundImage = `url(${data.image_data_x})`;
    });

  }).catch(function(error) {
    console.log(error);
  });

  var sliders = document.getElementsByClassName("slider");
  var outputs = [];
  for (var i = 0; i < sliders.length; i++){
    var label_text = sliders[i].previousElementSibling
    var output = sliders[i].nextElementSibling
    label_text.innerHTML = sliders[i].id[0].toUpperCase() + sliders[i].id.slice(1);
    output.innerHTML = sliders[i].value
    sliders[i].oninput = function() {
      config_update = layer_selection;
      var v = this.value;
      configs[layer_selection][this.id] = parseFloat(v);
      if (this.id=='translateX'){
          configs[layer_selection]['translate'][0] = parseFloat(v)
      }
      if (this.id=='translateY'){
          configs[layer_selection]['translate'][1] = parseFloat(v)
      }
      renderController(this, v);
    }

  }
  // let cluster_dropdown = document.querySelector("#idx");
  let cluster_demo = document.querySelector("#clusterDemo");
  let demoContainer = document.querySelector("#demoContainer")
  let demoImages = []
  for (var idx = 0; idx < 7; idx++){
    var img = document.createElement('img')
    img.setAttribute('class','clusterDemoImg')
    img.setAttribute('alt', '')
    demoImages.push(img)
    demoContainer.appendChild(img)
  }
  // cluster_dropdown.onchange = function () {
  //   updateClusterDemo()
  // };
  var clusterCheckboxDiv = document.getElementById("clusterLabels");
  let numClusterIn = document.querySelector("#clusterNum");
  var clusterCheckbox = [];
  function renderClusterCheckBox(){
      clusterCheckbox = [];
      clusterCheckboxDiv.replaceChildren()
      // cluster_dropdown.replaceChildren()
      // var option = document.createElement('option');
      // option.setAttribute('value',-1)
      // option.innerHTML = 'All'
      // cluster_dropdown.append(option)
      for (var i = 0; i < cluster_numbers[layer_selection]; i++){
        var div = document.createElement('div');
        var span = document.createElement('span');
        // span.innerHTML = "" + i + "&nbsp&nbsp&nbsp&nbsp"
        span.innerHTML = "&nbsp&nbspRoute"
        var inputdiv = document.createElement('input');
        inputdiv.setAttribute('type', 'checkbox')
        inputdiv.setAttribute('value', i)
        // inputdiv.checked = true;
        // span.setAttribute('class', 'textCheckBox boxChecked')

        div.setAttribute('class', 'textCheckBoxContainer')
        div.appendChild(inputdiv)
        div.appendChild(span)
        clusterCheckboxDiv.appendChild(div)
        clusterCheckbox.push(inputdiv)

        if (configs[layer_selection]['cluster'].includes(i)){
          clusterCheckbox[i].checked = true
          span.setAttribute('class', 'textCheckBox boxChecked')
        } else {
          clusterCheckbox[i].checked = false
          span.setAttribute('class', 'textCheckBox boxUnchecked')
        }

        span.addEventListener('click', function(){
          this.previousElementSibling.checked = !this.previousElementSibling.checked
          if (this.previousElementSibling.checked){
            this.setAttribute('class', 'textCheckBox boxChecked')
          } else {
            this.setAttribute('class', 'textCheckBox boxUnchecked')
          }
          config_update = layer_selection;
          var selection = [];
          for (var x = 0; x < clusterCheckbox.length; x++){
            if (clusterCheckbox[x].checked){
              selection.push(x)
            }
          }
          configs[layer_selection]['cluster'] = selection;
        })

        // var option = document.createElement('option');
        // option.setAttribute('value',i)
        // option.innerHTML = i
        // cluster_dropdown.append(option)
      }

      // for (var i = 0; i < clusterCheckbox.length; i++){
      //   if (configs[layer_selection]['cluster'].includes(i)){
      //     clusterCheckbox[i].checked = true
      //   } else {
      //     clusterCheckbox[i].checked = false
      //   }
      // }

  }
  renderClusterCheckBox();

  function renderController(slider, v){
    // output = slider.previousElementSibling
    // output.innerHTML = slider.id[0].toUpperCase() + slider.id.slice(1) + ": " + v;
    var output = slider.nextElementSibling;
    output.innerHTML = v
  }

  let clustersMainTd = document.querySelector("#clusters");

  function updateSelection(name){
    // console.log('select: '+name)
    // console.log(layer_list)
    layer_selection = name;

    renderControlTable();

    var keys = Object.keys(layer_list)
    console.log(keys)
    for (var i = 0; i < keys.length; i++){
      layer_list[keys[i]][1].setAttribute('class', 'layerNamesDiv layerDisSelected')
      if (layer_list[keys[i]][0]){
        layer_list[keys[i]][1].firstChild.setAttribute('class', 'layerSelectionBtn layerSelectionBtnDown')
      } else {
        layer_list[keys[i]][1].firstChild.setAttribute('class', 'layerSelectionBtn layerSelectionBtnUp')
      }
    }
    layer_list[name][1].setAttribute('class', 'layerNamesDiv layerSelected')
    if (layer_list[name][0]) {
      clustersMainTd.setAttribute('class', '')
    } else {
      clustersMainTd.setAttribute('class', 'greyMask')
    }
    updateClusterDemo()
  }


  function switchLayerActivation(name){
    // console.log(name + " switch")
    // console.log(layer_list)
    if (layer_list[name][0]){
      layer_list[name][0] = false
      // var this_cluster = [...Array(cluster_numbers[name]).keys()]
      // var this_config = {...configs_template}
      // this_config['cluster'] = this_cluster
      socket.emit('config_clear', layer_selection);
      console.log(name + ' cleared')
    } else {
      layer_list[name][0] = true
    }
    // layer_list[name][0] = !layer_list[name][0]
    config_update = name
  }

  function renderControlTable(){
    for (var i = 0; i < sliders.length; i++){
      var v = configs[layer_selection][sliders[i].id]
      sliders[i].value = v;
      var output = sliders[i].nextElementSibling;
      output.innerHTML = v
      renderController(sliders[i], v);
    }
    renderClusterCheckBox();

  }




  function updateClusterDemo(){
    let dataURL = canvas.toDataURL('image/jpeg',input_quality);
    // socket.emit('change_cluster_demo', cluster_dropdown.value, layer_selection, dataURL)
    socket.emit('change_cluster_demo', layer_selection, cluster_numbers[layer_selection], dataURL)
  }

  socket.on('return_cluster_demo',function(data){
    // cluster_demo.setAttribute('src', data.image_data);
    // // TODO:
    var keys = Object.keys(data)
    // console.log(keys)
    // console.log(demoImages)
    for (var i = 0; i < 7; i++){
      // console.log(demoImages[i])
      if (i<keys.length){
        demoImages[i].setAttribute('src', data[keys[i]]);
      } else {
        demoImages[i].setAttribute('src', './static/blank.png');
      }
    }
  });

  let generateClustersBtn = document.querySelector("#generateClusters");
  generateClustersBtn.onclick = function () {
    let dataURL = canvas.toDataURL('image/jpeg',input_quality);
    // socket.emit('regenerate_cluster', layer_selection, dataURL, numClusterIn.value, cluster_dropdown.value)
    socket.emit('regenerate_cluster', layer_selection, dataURL, numClusterIn.value, 0)
    console.log(numClusterIn.value)
    cluster_numbers[layer_selection] = parseInt(numClusterIn.value)
    resetLayer()
  };


  let resetLayerBtn = document.querySelector("#resetLayer");
  let resetAllBtn = document.querySelector("#resetAll");
  resetLayerBtn.onclick = function() {
    resetLayer()
  }
  function resetLayer() {
    config_update = layer_selection;
    configs[layer_selection] = {...configs_template};
    configs[layer_selection]['translate'] = [configs[layer_selection]["translateX"], configs[layer_selection]["translateY"]];
    configs[layer_selection]['cluster'] = [...Array(cluster_numbers[layer_selection]).keys()]
    renderControlTable();
  }
  // resetAllBtn.onclick = function() {
  //   config_update = layer_selection;
  //   var keys = Object.keys(layer_list)
  //   for (var i = 0; i < keys.length; i++){
  //     configs[keys[i]] = {...configs_template};
  //     configs[keys[i]]['translate'] = [configs[keys[i]]["translateX"], configs[keys[i]]["translateY"]];
  //   }
  //   renderControlTable();
  // }

});
