import React, { useState, useEffect } from "react";
import "./sidebar.css";
import SidebarButton from "./sidebarButton";
import { MdFavorite } from "react-icons/md";
import { FaGripfire, FaPlay } from "react-icons/fa";
import { FaSignOutAlt } from "react-icons/fa";
import { IoLibrary } from "react-icons/io5";
import { MdSpaceDashboard } from "react-icons/md";
import apiClient from "../../spotify";
import { IconContext } from "react-icons";
export default function Sidebar() {
  const [image, setImage] = useState(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdLAY3C19kL0nV2bI_plU3_YFCtra0dpsYkg&usqp=CAU"
  );
  useEffect(() => {
    apiClient.get("me").then((response) => {
      setImage(response.data.images[0].url);
      // console.log(response);
    });
  }, []);
  return (
    <div className="sidebar-container">
      <img src={image} className="profile-img" alt="profile" />
      <div className="feedandlibrary">
        <SidebarButton title="Feed" to="/feed" icon={<MdSpaceDashboard />} />
        {/* <SidebarButton title="Trending" to="/trending" icon={<FaGripfire />} />
        <SidebarButton title="Player" to="/player" icon={<FaPlay />} />
        <SidebarButton
          title="Favorites"
          to="/favorites"
          icon={<MdFavorite />}
        /> */}
        <SidebarButton title="Library" to="/" icon={<IoLibrary />} />
      </div>


      <div className="btn-body" onClick={() => {
        localStorage.removeItem("token");
        window.location.reload();
      }}>
        <IconContext.Provider value={{ size: "24px", className: "btn-icon" }}>
          <FaSignOutAlt />
          <p className="btn-title">SignOut</p>
        </IconContext.Provider>
      </div>

    </div>
  );
}
