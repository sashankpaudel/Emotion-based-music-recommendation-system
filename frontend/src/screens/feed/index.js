import React, { useState, useEffect, useRef } from 'react';
import APIKit from "../../spotify";
import { IconContext } from "react-icons";
import { AiFillPlayCircle } from "react-icons/ai";
import axios from 'axios';
import "./feed.css";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";


export default function Feed() {
  const videoRef = useRef(null);
  const webRef = useRef(null);
  
  const [feed, setFeed] = useState()
  const [img, setImg] = useState()
  const [playlists, setPlaylists] = useState(null);
  const [imageSrc, setImageSrc] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [captureLoading, setCaptureLoading] = useState(false);
  const [src, setSrc] = useState();
  const [err, setErr] = useState(null);
  const navigate = useNavigate();
  const showImage = () => {
    //  const formData= new FormData()  
    console.log('click')
    const imageSrc = webRef.current.getScreenshot()
    setImageSrc(imageSrc)
    setCaptureLoading(true)

    // Send the image to the backend
    axios.post('http://127.0.0.1:8000/webcam/', { image: imageSrc.split(',')[1] })
      .then(response => {
        console.log(response)
        // Handle response from backend
        if (response.data.message === 'No emotion detected') {
          setResponse(null)
        }
        else { setResponse(response.data.message); }
        setCaptureLoading(false)
      })
      .catch(error => {
        // Handle error
        console.error('Error sending image:', error);
      });
  };



  function getBase64 (file, callback) {

    const reader = new FileReader();

    reader.addEventListener('load', () => callback(reader.result));

    reader.readAsDataURL(file);
}
  const handleImageChange = (e) => {
    setSrc(URL.createObjectURL(e.target.files[0]))

    const img= e.target.files[0]

   const image= getBase64(img, function(base64Data){
     axios.post('http://127.0.0.1:8000/webcam/',{image:base64Data.split(',')[1]})
       .then(response => {
         console.log(response)
         // Handle response from backend
         if (response.data.message === 'No emotion detected') {
           setResponse(null)
         }
         else { setResponse(response.data.message); }
         setCaptureLoading(false)
       })
       .catch(error => {
         // Handle error
         console.error('Error sending image:', error);
       });
  });


    // Send the image to the backend
  };



  // const token = localStorage.getItem('token')

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/getsong").then(res => {
      setFeed(JSON.parse(res.data.message));
      console.log(JSON.parse(res.data.message))
    })
  }, [])

  console.log(feed)

  const list = feed && Object.entries(feed?.url)?.map((itm) => itm[1])
  const name = feed && Object.entries(feed?.artists_song)?.map((itm) => itm[1])
  const id = feed && Object.entries(feed?.id)?.map((itm) => itm[1])
  console.log(list)

  const onRecommend = () => {
    setLoading(true)
    axios.post("http://127.0.0.1:8000/getsong/", { response }).then(res => {
      setFeed(JSON.parse(res.data.message));
      setLoading(false)
      console.log(JSON.parse(res.data.message))
    }).catch(err => {
      setErr(err)
      console.log(err)
    })
  }
  return (
    <div className='feed-container'>
      <Webcam className='webcam' ref={webRef} />
      {/* <input type="file" onChange={(e)=>handleImageChange(e)}/> */}
      <button className='button' onClick={() => showImage()}>Capture 📷</button>{ }
      {src&&<img src={ src} alt="" />}
      {/* <video ref={videoRef} autoPlay></video> */}
      {imageSrc && <img src={imageSrc} alt="" />}
      {captureLoading ? <p className='texts'>Loading...</p> :response? <p className='texts'>You look {response}</p>:<p className='texts'>No emotion detected</p>}


      <button className='button' onClick={onRecommend} disabled={!response || response==='No emotion detected'? true: false}>Recommend</button>
      {loading ? <p className='texts'>Recommending, please wait...</p> :
        <div className='labels'>
          {response && response === 'No emotion detected' ? null : <>
            <ul className='list-song'>

              {list?.map((img, i) => <li onClick={()=>window.open(`https://open.spotify.com/track/${id[i]}`)}>
                <img className="albumImage" src={img} alt="" />
                <h5 className='labelName'>   {name[i]}</h5>

              </li>)}
            </ul>
          </>}
          </div>
      }

    </div>
  );
};

