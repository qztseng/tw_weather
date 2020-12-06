L.Control.SliderControl = L.Control.extend({
    options: {
        position: 'bottomleft',
        layerGroups: null,
        timeAttribute: 'time',
        isEpoch: false,     // whether the time attribute is seconds elapsed from epoch
        startTimeIdx: 0,    // where to start looking for a timestring
        timeStrLength: 19,  // the size of  yyyy-mm-dd hh:mm:ss - if millis are present this will be larger
        maxValue: -1,
        minValue: 0,
        showAllOnStart: false,
        range: false,
        follow: false,
        alwaysShowDate : false,
    },

    initialize: function (options) {
        L.Util.setOptions(this, options);
        this._layerGroups = this.options.layerGroups;
    },

    extractTimestamp: function(time, options) {
        if (options.isEpoch) {
            time = (new Date(parseInt(time))).toString(); // this is local time
        }
        return time.substr(options.startTimeIdx, options.startTimeIdx + options.timeStrLength);
    },

    setPosition: function (position) {
        var map = this._map;

        if (map) {
            map.removeControl(this);
        }

        this.options.position = position;

        if (map) {
            map.addControl(this);
        }
        this.startSlider();
        return this;
    },

    onAdd: function (map) {
        this.options.map = map;

        // Create a control sliderContainer with a jquery ui slider
        var sliderContainer = L.DomUtil.create('div', 'slider', this._container);
        $(sliderContainer).append('<div id="leaflet-slider" style="width:200px"><div class="ui-slider-handle"></div><div id="slider-timestamp" style="width:200px; margin-top:13px; background-color:#FFFFFF; text-align:center; border-radius:5px;"></div></div>');
        //Prevent map panning/zooming while using the slider
        $(sliderContainer).mousedown(function () {
            map.dragging.disable();
        });
        $(document).mouseup(function () {
            map.dragging.enable();
            //Hide the slider timestamp if not range and option alwaysShowDate is set on false
            // if (options.range || !options.alwaysShowDate) {
            //     $('#slider-timestamp').html('');
            // }
        });

        this.easybar = this._createEasyBar(map);

        var options = this.options;
        this.options.layers = [];

        //If a layer has been provided: calculate the min and max values for the slider
        if (this._layerGroups) {
            var index_temp = 0;
            for (l in this._layerGroups) {
                options.layers[index_temp] = this._layerGroups[l];
                ++index_temp;
            }
            options.maxValue = index_temp - 1;
            this.options = options;
        } else {
            console.log("Error: You have to specify a layerGroups via new SliderControl({layer: your_layer});");
        }

        //move slider to current month on startup
        // hs=$('#leaflet-slider').slider();
        // sv=10
        // hs.slider('option', 'value',sv);
        // hs.slider('option','slide').call(hs,null,{ handle: $('.ui-slider-handle', hs), value: sv });

        return sliderContainer;
    },


    _createEasyBar: function(map) {
       
        bwdB = L.easyButton('fa-backward', function(){stepBackward()}, title='bwd', id=0);
        stopB = L.easyButton('fa-stop', function(){stopButton()}, title='stop', id=1);
        playB = L.easyButton('fa-play', function(){playButton()}, title='play', id=2);
        fwdB = L.easyButton('fa-forward', function(){stepForward()}, title='fwd', id=3);

        var easyBar = L.easyBar([bwdB, stopB, playB, fwdB]);
        easyBar.options.position = 'bottomleft';
        easyBar.addTo(map);

        var play = false

        var stopButton = function f(){
           play = false
        }

        var playButton = async function f(){
            
            const timer = ms => new Promise(res => setTimeout(res, ms));

            play = true;
            while (play) {
                stepForward();
                await timer(500); 
            }
        }

        var stepForward = function(){
            
            hs=$('#leaflet-slider').slider();
            sv = hs.slider("option", "value") + 1;
            max = hs.slider("option", "max");
            min = hs.slider("option", "min");


            if(sv>max){
                sv=min;
            }
            hs.slider('option', 'value',sv);
            hs.slider('option','slide')
                .call(hs,null,{ handle: $('.ui-slider-handle', hs), value: sv });

        }

        var stepBackward = function(){
            
            hs=$('#leaflet-slider').slider();
            sv = hs.slider("option", "value") - 1;
            max = hs.slider("option", "max");
            min = hs.slider("option", "min");


            if(sv<min){
                sv=max;
            }
            hs.slider('option', 'value',sv);
            hs.slider('option','slide')
                .call(hs,null,{ handle: $('.ui-slider-handle', hs), value: sv });

        }

        return easyBar
    },


    onRemove: function (map) {
        //Delete all layergroups which where added via the slider and remove the slider div
        for (i = this.options.minValue; i <= this.options.maxValue; i++) {
            map.removeLayer(this.options.layers[i]);
        }
        $('#leaflet-slider').remove();

        // unbind listeners to prevent memory leaks
        $(document).off("mouseup");
        $(".slider").off("mousedown");
    },


    startSlider: function () {
        _options = this.options;
        _extractTimeStamp = this.extractTimestamp;
        var index_start = _options.minValue;
        if(_options.showAllOnStart){
            index_start = _options.maxValue;
            if(_options.range) _options.values = [_options.minValue,_options.maxValue];
            else _options.value = _options.maxValue;
        } else {
            for (i in _options.layers){
                _options.layers[i].remove();
            }
        }
        $("#leaflet-slider").slider({
            range: _options.range,
            value: _options.value,
            values: _options.values,
            min: _options.minValue,
            max: _options.maxValue,
            step: 1,
            slide: function (e, ui) {
                var map = _options.map;
                // var fg = L.featureGroup();
                if(!!_options.layers[ui.value]) {
                    // If there is no time property, this line has to be removed (or exchanged with a different property)
                    if(_options.layers[ui.value].feature !== undefined) {
                        // console.log(_options.layers[ui.value])
                        // if(_options.markers[ui.value].feature.properties[_options.timeAttribute]){
                        //     if(_options.markers[ui.value]) $('#slider-timestamp').html(
                        //         _extractTimeStamp(_options.markers[ui.value].feature.properties[_options.timeAttribute], _options));
                        // }else {
                        //     console.error("Time property "+ _options.timeAttribute +" not found in data");
                        // }
                    }else {
                        // set by leaflet Vector Layers
                        if(_options.layers[ui.value].options[_options.timeAttribute]){
                            if(_options.layers[ui.value]) $('#slider-timestamp').html(
                                _extractTimeStamp(_options.layers[ui.value].options[_options.timeAttribute], _options));
                        }else {
                            console.error("Time property "+ _options.timeAttribute +" not found in data");
                        }
                    }

                    var i;
                    // clear layergroups
                    for (i = _options.minValue; i <= _options.maxValue; i++) {
                        if(_options.layers[i]) map.removeLayer(_options.layers[i]);
                    }
                    if(_options.range){
                        // jquery ui using range
                        for (i = ui.values[0]; i <= ui.values[1]; i++){
                           if(_options.layers[i]) {
                               map.addLayer(_options.layers[i]);
                           }
                        }
                    }else if(_options.follow){
                        for (i = ui.value - _options.follow + 1; i <= ui.value ; i++) {
                            if(_options.layers[i]) {
                                map.addLayer(_options.layers[i]);
                            }
                        }
                    }else{
                        for (i = _options.minValue; i <= ui.value ; i++) {
                            if(_options.layers[i]) {
                                map.addLayer(_options.layers[i]);
                            }
                        }
                    }
                };
            }
        });
        if (!_options.range && _options.alwaysShowDate) {
            // $('#slider-timestamp').html(_extractTimeStamp(_options.layers[index_start].feature.properties[_options.timeAttribute], _options));
        }
        for (i = _options.minValue; i <= index_start; i++) {
            // _options.map.addLayer(_options.layers[i]);
            _options.layers[i].addTo(_options.map);
        }

        // move slider to current month on startup
        d = new Date();
        sv = d.getMonth();
        $("#leaflet-slider").slider('option', 'value',sv);
        $('#slider-timestamp').html(_extractTimeStamp(_options.layers[sv].options[_options.timeAttribute], _options));
    }
});


L.control.sliderControl = function (options) {
    return new L.Control.SliderControl(options);
};
