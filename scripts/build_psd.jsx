// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 ChitoseRen
/* Bounded Photoshop assembler. Set SEMANTIC_PSD_REQUEST before $.evalFile().
   begin: {phase:'begin', name, canvas:[width,height], resolution:72}
   add: {phase:'add', document_id, layers:[{file:absolutePNG,name,visible:true}]}
   finish: {phase:'finish',document_id,psd:absolutePSD,preview:absolutePNG}
   inspect: {phase:'inspect',document_id}
   add accepts <=12 layers, in back-to-front order. All PNGs must share canvas.
   Returns a JSON string; does not merge layers or close the target document.
*/
(function (q) {
    function quote(s) { return '"' + String(s).replace(/\\/g,'\\\\').replace(/"/g,'\\"').replace(/\r/g,'\\r').replace(/\n/g,'\\n') + '"'; }
    function encode(x) {
        if (x === null) return 'null';
        if (typeof x === 'string') return quote(x);
        if (typeof x === 'number' || typeof x === 'boolean') return String(x);
        var a=[], k;
        if (x instanceof Array) { for(k=0;k<x.length;k++) a.push(encode(x[k]));return '['+a.join(',')+']'; }
        for(k in x) if(x.hasOwnProperty(k)) a.push(quote(k)+':'+encode(x[k]));
        return '{'+a.join(',')+'}';
    }
    function target(id) {
        for(var i=0;i<app.documents.length;i++) if(app.documents[i].id===id) { app.activeDocument=app.documents[i];return app.documents[i]; }
        throw new Error('Target document is not open: '+id);
    }
    function info(d) {
        var layers=[];
        for(var i=0;i<d.layers.length;i++) layers.push({name:d.layers[i].name,visible:d.layers[i].visible,kind:d.layers[i].typename});
        return {document_id:d.id,name:d.name,canvas:[d.width.as('px'),d.height.as('px')],layers_top_to_bottom:layers};
    }
    function outputFile(path, extension) {
        var f=new File(path);
        if(!f.parent.exists) throw new Error('Output folder must already exist: '+f.parent.fsName);
        if(f.name.toLowerCase().slice(-extension.length)!==extension) throw new Error('Wrong output extension: '+f.name);
        return f;
    }
    var oldDialogs=app.displayDialogs, oldUnits=app.preferences.rulerUnits;
    app.displayDialogs=DialogModes.NO;app.preferences.rulerUnits=Units.PIXELS;
    try {
        if(!q || !q.phase) throw new Error('SEMANTIC_PSD_REQUEST.phase required');
        var d, i;
        if(q.phase==='begin') {
            if(!q.canvas || q.canvas.length!==2 || q.canvas[0]<=0 || q.canvas[1]<=0) throw new Error('Positive canvas dimensions required');
            d=app.documents.add(UnitValue(q.canvas[0],'px'),UnitValue(q.canvas[1],'px'),q.resolution||72,q.name||'Semantic illustration',NewDocumentMode.RGB,DocumentFill.TRANSPARENT,1,BitsPerChannelType.EIGHT);
            d.activeLayer.name='__assembly_empty__';
        } else {
            d=target(q.document_id);
            if(q.phase==='add') {
                if(!q.layers || !q.layers.length || q.layers.length>12) throw new Error('add requires 1-12 layers');
                // Validate every path before changing the document.
                for(i=0;i<q.layers.length;i++) {
                    var input=new File(q.layers[i].file);
                    if(!input.exists || !/\.png$/i.test(input.name)) throw new Error('Existing PNG required: '+input.fsName);
                    if(!q.layers[i].name) throw new Error('Explicit layer name required');
                    for(var z=0;z<d.layers.length;z++) if(d.layers[z].name===q.layers[i].name) throw new Error('Duplicate name; inspect before retrying: '+q.layers[i].name);
                }
                for(i=0;i<q.layers.length;i++) {
                    var source=app.open(new File(q.layers[i].file)), copy;
                    try {
                        // Different document DPI can produce sub-pixel floating-point noise for the same integer raster size.
                        if(Math.round(source.width.as('px'))!==Math.round(d.width.as('px')) || Math.round(source.height.as('px'))!==Math.round(d.height.as('px'))) throw new Error('PNG must match full canvas: '+q.layers[i].file);
                        if(source.layers.length!==1) throw new Error('PNG must contain one pixel layer');
                        copy=source.activeLayer.duplicate(d,ElementPlacement.PLACEATBEGINNING);
                    } finally { source.close(SaveOptions.DONOTSAVECHANGES);app.activeDocument=d; }
                    // Photoshop can reject opacity/blend changes after hiding the active layer.
                    copy.name=q.layers[i].name;copy.visible=true;copy.opacity=100;copy.blendMode=BlendMode.NORMAL;copy.visible=q.layers[i].visible!==false;
                }
                // Refresh layer handles after imports before applying final visibility.
                for(i=0;i<q.layers.length;i++) {
                    var imported=d.artLayers.getByName(q.layers[i].name);
                    imported.visible=q.layers[i].visible!==false;
                }
                for(i=0;i<d.layers.length;i++) if(d.layers[i].visible) { d.activeLayer=d.layers[i];break; }
                for(i=d.layers.length-1;i>=0;i--) if(d.layers[i].name==='__assembly_empty__' && d.layers.length>1) d.layers[i].remove();
            } else if(q.phase==='finish') {
                var psd=outputFile(q.psd,'.psd'), png=outputFile(q.preview,'.png');
                var options=new PhotoshopSaveOptions();options.layers=true;options.embedColorProfile=true;options.alphaChannels=true;
                d.saveAs(psd,options,false,Extension.LOWERCASE);
                var exportOptions=new ExportOptionsSaveForWeb();exportOptions.format=SaveDocumentType.PNG;exportOptions.PNG8=false;exportOptions.transparency=true;exportOptions.interlaced=false;
                d.exportDocument(png,ExportType.SAVEFORWEB,exportOptions);
            } else if(q.phase!=='inspect') throw new Error('Unknown phase: '+q.phase);
        }
        return encode({ok:true,phase:q.phase,document:info(d)});
    } catch(e) { return encode({ok:false,error:String(e),line:e.line||null,phase:q?q.phase:null}); }
    finally { app.displayDialogs=oldDialogs;app.preferences.rulerUnits=oldUnits; }
})(typeof SEMANTIC_PSD_REQUEST === 'undefined' ? null : SEMANTIC_PSD_REQUEST);
