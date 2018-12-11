class ResultData {
	public:
		enum DISTORTION_TYPE { NOISE, BLUR };

	private:
		DISTORTION_TYPE distortion_type;

	public:
    	ResultData(DISTORTION_TYPE distortion_type);
};