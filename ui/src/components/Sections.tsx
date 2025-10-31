import { Button, Modal, Snackbar, Typography } from '@mui/material';
import { Box } from '@mui/system';
import React, { useCallback } from 'react';
import { SectionsInner } from './report/SectionsInner';
import { MessageFeedbackComp } from './widgets/MessageFeedback';
import { Sections as SectionsType } from '../@types/AsyncTaskState';


interface PropType {
  sections: SectionsType;
  taskId: string;
  cookieUserId: string;
  reportTitle?: string | null;
}

export const Sections: React.FC<PropType> = (props) => {
  const { sections, taskId, cookieUserId, reportTitle } = props

  const [open, setOpen] = React.useState(false);
  const [openShare, setOpenShare] = React.useState(false);

  const handleModalOpen = useCallback(() => setOpen(true), [setOpen]);
  const handleModalClose = useCallback(() => setOpen(false), [setOpen]);
  const handleShare = useCallback(() => {
    navigator.clipboard.writeText(window.location.href);
    setOpenShare(true);
  }, []);
  const handleCloseShare = useCallback(() => setOpenShare(false), [setOpenShare]);


  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: '12px'}}>
          <Typography sx={{ fontSize:'14px', fontStyle: 'italic', opacity: '0.8', marginLeft:'6px' }}>Are these answers helpful?</Typography>
          <MessageFeedbackComp darkMode taskId={taskId} userId={cookieUserId} section={null} />
        </Box>

        <Box sx={{ display: 'flex' }}>
          <Button onClick={handleShare}>Share</Button>
          <Button onClick={handleModalOpen}>Disclaimer</Button>
        </Box>
      </Box>

      {reportTitle && (
        <Typography variant="h5" sx={{ marginBottom: '24px', marginTop: '8px', fontWeight: 500, color: 'white' }}>
          {reportTitle}
        </Typography>
      )}

      <SectionsInner sections={sections} taskId={taskId} cookieUserId={cookieUserId} />

      <Modal open={open} onClose={handleModalClose}>
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: { xs: '90vw', sm: '60vw' }, bgcolor: 'background.paper', boxShadow: 24, padding: { xs: '16px', sm: '28px 32px' } }}>
          <Typography variant="h4" sx={{ marginBottom: '12px' }}>Disclaimer</Typography>
          <Typography variant="body1" sx={{ marginBottom: '8px' }}>Ai2 Scholar QA answers questions by retrieving open-access papers from the scientific literature. It is not designed to answer non-scientific questions or questions that require sources outside the scientific literature.</Typography>
          <Typography variant="body1">Its output may have errors, and these errors might be difficult to detect. For example, there might be serious factual inaccuracies or omissions. Please verify the accuracy of the generated text whenever possible.</Typography>
        </Box>
      </Modal>

      <Snackbar
        open={openShare}
        autoHideDuration={1000}
        onClose={handleCloseShare}
        message="Link copied to clipboard"
      />

    </Box>
  );
};
